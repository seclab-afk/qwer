from ast import List
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, StorageContext, load_index_from_storage, Settings, QueryBundle, Document, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, TokenCountingHandler, CBEventType, EventPayload
from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.prompts import RichPromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Tuple, Dict, Any, Optional
import os
import subprocess
import yaml
import logging
import shutil
import glob
import docker
import socket
import time
import re
import json
import asyncio
import httpx
import datetime
from config import HOST_BASE_PATH
from logger_config import get_logger, get_perf_log_file, get_prompt_log_file, get_token_file
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi

logger = None

llm_semaphore = asyncio.Semaphore(1)

_coverage_config_cache = None

def save_performance_detailed_log(task: str, lib: str, category: str, input_len: int, total_time: float, rag_time: float, llm_time: float, in_tokens: int = 0, out_tokens: int = 0):
    """Save performance data to a unique log file per execution."""
    perf_file = get_perf_log_file()
    log_dir = os.path.dirname(perf_file)
    
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}] TASK={task:15} | LIB={lib:10} | CAT={category:15} | "
        f"TOTAL={total_time:7.2f}s | RAG={rag_time:7.2f}s | LLM={llm_time:7.2f}s | "
        f"IN_TOK={in_tokens:8} | OUT_TOK={out_tokens:7}\n"
    )
    try:
        with open(perf_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass

def save_prompt_log(task: str, lib: str, category: str, prompt: str):
    """Save the full prompt passed to LLM to a separate log file."""
    if not prompt:
        return
        
    prompt_file = get_prompt_log_file()
    log_dir = os.path.dirname(prompt_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n{'='*80}\n[{timestamp}] TASK={task} | LIB={lib} | CAT={category}\n{'='*80}\n"
    footer = f"\n{'='*80}\n"
    
    try:
        with open(prompt_file, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(prompt)
            f.write(footer)
    except Exception:
        pass

class PromptLoggerHandler(BaseCallbackHandler):
    """Callback handler to intercept and log the prompt sent to LLM."""
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.current_task = "UNKNOWN"
        self.current_lib = "UNKNOWN"
        self.current_category = "N/A"

    def on_event_start(self, event_type, payload, event_id, **kwargs):
        if event_type == CBEventType.LLM:
            prompt = payload.get(EventPayload.PROMPT)
            if not prompt:
                messages = payload.get(EventPayload.MESSAGES)
                if messages:
                    prompt = "\n".join([f"[{getattr(m, 'role', 'user')}]: {getattr(m, 'content', str(m))}" for m in messages])
            
            if prompt:
                save_prompt_log(f"PRE_{self.current_task}", self.current_lib, self.current_category, prompt)

    def on_event_end(self, event_type, payload, event_id, **kwargs):
        if event_type == CBEventType.LLM:
            response = payload.get(EventPayload.RESPONSE)
            if response:
                response_text = ""
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    response_text = response.message.content
                else:
                    response_text = str(response)
                
                if response_text:
                    save_prompt_log(f"POST_{self.current_task}", self.current_lib, self.current_category, response_text)

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass

def load_coverage_config() -> dict:
    """
    Load library-specific coverage exclusion patterns from coverage_config.yaml.
    Caches the result to prevent repeated loading.
    """
    global _coverage_config_cache
    if _coverage_config_cache is not None:
        return _coverage_config_cache
    
    config_path = os.path.join(os.path.dirname(__file__), "coverage_config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _coverage_config_cache = yaml.safe_load(f) or {}
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load coverage_config.yaml: {e}")
        _coverage_config_cache = {}
    
    return _coverage_config_cache

def get_coverage_exclude_patterns(library_name: str) -> list:
    """
    Return coverage exclusion pattern list for the library.
    Harness file exclusion pattern is always included.
    """
    config = load_coverage_config()
    library_config = config.get(library_name) or {}
    exclude_patterns = library_config.get("exclude_patterns") or []
    
    base_patterns = ["_harness"]

    return base_patterns + list(exclude_patterns)

def build_coverage_ignore_regex(library_name: str) -> str:
    """
    Build --ignore-filename-regex flag string for llvm-cov.
    """
    patterns = get_coverage_exclude_patterns(library_name)
    if not patterns:
        return ""
    
    regex_parts = "|".join([f".*{p}.*" for p in patterns])
    return f'-ignore-filename-regex="({regex_parts})"'

def get_unused_apis(lib: str, all_apis: list) -> list:
    """
    Return list of APIs not yet used in harnesses.
    Used as feedback for LLM during re-categorization/flow generation.
    
    Args:
        lib: Library name
        all_apis: Complete API list loaded from extracted_api.txt
    
    Returns:
        List of unused APIs (empty list if none)
    """
    harness_dir = f"library/{lib}/final_harnesses"
    
    if not os.path.exists(harness_dir):
        return all_apis
    
    try:
        all_harness_content = ""
        for root, dirs, files in os.walk(harness_dir):
            for file in files:
                if file.endswith('.c'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            all_harness_content += f.read() + "\n"
                    except Exception:
                        continue
        
        if not all_harness_content:
            return all_apis
        

        unused_apis = []
        for api in all_apis:
            if api not in all_harness_content:
                unused_apis.append(api)
        
        if logger:
            logger.info(f"[get_unused_apis] {len(unused_apis)} unused out of {len(all_apis)} APIs")
        
        return unused_apis
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to calculate unused APIs: {e}")
        return []

class OllamaRAGManager:
    """Class for RAG management"""
    def __init__(self, lib, model="", base_url="http://localhost:11434", embedding_model_name="BAAI/bge-small-en-v1.5", context_window=60000, token_callback=None):
        self.lib = lib
        self.lib_dir = os.path.join("library", lib, f"{lib}_api")
        self.rag_store=os.path.join("rag_store", lib)
        os.makedirs(self.rag_store, exist_ok=True)
        self.index_dir = os.path.join(self.rag_store, f"{lib}_index_rag")
        self.prototype_dir = os.path.join(self.rag_store, f"{lib}_prototype")
        self.lib_docs_dir = os.path.join("library", lib, f"{lib}_docs")
        self.file_exts = [".h", ".c", ".cc", ".cpp", ".txt", ".md", ".rst", ".pdf"]
        self.llm = OpenAILike(
            model=model,
            api_base=f"{base_url}/v1",
            api_key="EMPTY",
            is_chat_model=True,
            temperature=0.9,
            context_window=context_window,
            timeout=600.0,
        )
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        self.index = None
        self.query_engine = {}
        self.chat_engines = {}
        self.absolute_lib_dir =""
        self.prompt_logger = PromptLoggerHandler()
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        self.token_counter = TokenCountingHandler(verbose=False)
        callback_manager = CallbackManager([self.llama_debug, self.prompt_logger, self.token_counter])
        Settings.callback_manager = callback_manager
        self.token_callback = token_callback

    @staticmethod
    def collect_files(src_dir, dst_dir, exts):
        os.makedirs(dst_dir, exist_ok=True)
        for root, _, files in os.walk(src_dir):
            for file in files:
                if any(file.endswith(ext) for ext in exts):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dst_dir, file)
                    if not os.path.exists(dst_path):
                        if file.endswith(".pdf"):
                            shutil.copy2(src_path, dst_path)
                        else:
                            with open(src_path, "r", encoding="utf-8", errors="ignore") as fsrc, \
                                 open(dst_path, "w", encoding="utf-8") as fdst:
                                fdst.write(fsrc.read())
                                    
    def extract_ast_rag(self, api_list:List, include_dirs:List):
        os.makedirs(os.path.join("library", self.lib, "json"), exist_ok=True)
        remaining_apis = api_list.copy()

        lib_dir = self.lib_dir
        self.absolute_lib_dir = os.path.dirname(os.path.abspath(lib_dir)) 
        lib_name=self.lib+"_api"
        
        def _dump_clang_ast(source_path: Path, include_dirs: List) -> Dict[str, Any]:
            suffix = source_path.suffix.lower()
            is_cpp = suffix in {".cc", ".cpp", ".cxx"}
            compiler = "clang++" if is_cpp else "clang"
            lang = "c++" if is_cpp else "c"
            
            inc = ""
            for i in include_dirs:
                inc += f"-I{i.replace('{lib_base}', f'./{self.lib_dir}')} "
            clang_cmd = f'{compiler} -x {lang} -fsyntax-only -Xclang -ast-dump=json {inc}{str(source_path)} > {os.path.join("library", self.lib, "json", Path(source_path.name).with_suffix(".json"))}'
            proc = subprocess.run(clang_cmd, shell=True, capture_output=True, text=True)
            return proc.returncode
        
        def _unwrap_clang_location(loc: Dict[str, Any]) -> (Optional[str], Optional[int]):
            if not isinstance(loc, dict):
                return None, None
            included = loc.get("includedFrom")
            if isinstance(included, dict) and included.get("file"):
                included_file = included.get("file")
                included_line = included.get("line") or loc.get("line")
                if included_file:
                    return included_file, included_line
            file_path = loc.get("file")
            line = loc.get("line")
            if file_path:
                return file_path, line
            for key in ("expansionLoc", "spellingLoc"):
                sub_file, sub_line = _unwrap_clang_location(loc.get(key))
                if sub_file:
                    return sub_file, sub_line
            return None, line
        
        def _has_function_body(node: Dict[str, Any]) -> bool:
            for child in node.get("inner", []) or []:
                if isinstance(child, dict) and child.get("kind") in {"CompoundStmt", "CXXTryStmt"}:
                    return True
            return False

        def _capture_leading_comment(lines: [str], start_idx: int) -> Optional[str]:
            if not lines or start_idx <= 0 or start_idx >= len(lines):
                return None
            idx = start_idx - 1
            while idx >= 0 and not lines[idx].strip():
                idx -= 1
            if idx < 0:
                return None
            stripped = lines[idx].lstrip()
            comments: [str] = []
            if stripped.startswith("//"):
                while idx >= 0 and lines[idx].lstrip().startswith("//"):
                    comments.append(lines[idx].rstrip())
                    idx -= 1
                comments.reverse()
                return "\n".join(comments)
            if "/*" in stripped or "*/" in stripped:
                while idx >= 0:
                    comments.append(lines[idx].rstrip())
                    if "/*" in lines[idx]:
                        break
                    idx -= 1
                else:
                    return None
                comments.reverse()
                block = "\n".join(comments)
                if "/*" in block and "*/" in block:
                    return block
            return None


        def _extract_function_chunks(target_path: str, lines: [str]) -> [Dict[str, Any]]:
            json_path = os.path.join("library", self.lib, "json", Path(target_path).with_suffix(".json").name)
            with open(json_path,'r') as f:
                ast_json = json.load(f)
            
            chunks = []
            
            def visit(node: Any) -> None:
                if isinstance(node, dict):
                    if node.get("kind") in {"FunctionDecl", "CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl"}:
                        node_name = node.get("name")
                        if node_name:
                            matched_api = None
                            for api in remaining_apis:
                                clean_api = api.split('(')[0].strip()
                                api_func_name = clean_api.split('::')[-1].strip()
                                if node_name == api_func_name:
                                    matched_api = api
                                    break
                            
                            if matched_api:
                                begin_file, begin_line = _unwrap_clang_location((node.get("range") or {}).get("begin", {}))
                                if begin_file is None and begin_line is not None:
                                    begin_file = target_path
                                if not begin_file:
                                    begin_file, begin_line = _unwrap_clang_location(node.get("loc", {}))
                                    if begin_file is None and begin_line is not None:
                                        begin_file = target_path
                                 
                                # If begin_file is still None, use target_path as fallback
                                if not begin_file:
                                    begin_file = target_path
                                    
                                if Path(begin_file).resolve() == Path(target_path):
                                    _, end_line = _unwrap_clang_location((node.get("range") or {}).get("end", {}))
                                    if end_line is None:
                                        _, end_line = _unwrap_clang_location(node.get("loc", {}))
                                    
                                    # Extract comment (optional)
                                    comment_text = node.get("rawComment")
                                    if not comment_text and begin_line is not None and lines:
                                        start_idx = max(begin_line - 1, 0)
                                        start_idx = min(start_idx, len(lines) - 1)
                                        comment_text = _capture_leading_comment(lines, start_idx)
                                    
                                    # Extract function type and return type
                                    func_type = node.get("type", {}).get("qualType", "")
                                    return_type = func_type.split("(", 1)[0].strip() if "(" in func_type else func_type
                                    
                                    # Remove from remaining_apis and append chunk
                                    remaining_apis.remove(matched_api)
                                    chunks.append(
                                        {
                                            "file_name": Path(target_path).name,
                                            "name": matched_api,
                                            "comment": comment_text,
                                            "params": [
                                                {
                                                    "name": param.get("name"),
                                                    "type": param.get("type", {}).get("qualType", ""),
                                                }
                                                for param in (node.get("inner") or [])
                                                if isinstance(param, dict) and param.get("kind") == "ParmVarDecl"
                                            ],
                                            "return_type": return_type,
                                        }
                                    )
                    for child in node.get("inner", []) or []:
                        visit(child)
                elif isinstance(node, (list, tuple)):
                    for item in node:
                        visit(item)

            visit(ast_json)
            return chunks
        def extract_functions_from_source(source_path, include_dirs: List) -> [Dict[str, Any]]:
            code = source_path.read_text(errors="ignore")
            lines = code.splitlines()
            _dump_clang_ast(source_path, include_dirs)
            return _extract_function_chunks(str(source_path.resolve()), lines)
            
        
        gadgets=[]

        full_library_dir = Path(self.absolute_lib_dir) / lib_name
        print(f"DBG) scanning: {full_library_dir}")
        
        if not include_dirs:
            include_dirs.append(f"./{self.lib_dir}")
        
        logger.info(f"[INCLUDE_DIRS]{include_dirs}")
        
        excluded_dir_keywords = {'test', 'tests', 'example', 'examples', 'demo', 'demos', 'benchmark', 'benchmarks', 'sample', 'samples', 'arm'}
        
        for source_file in full_library_dir.rglob("*"):
            if not remaining_apis:  
                break
            if not source_file.is_file():
                continue
            
            if any(keyword in part.lower() for part in source_file.parts for keyword in excluded_dir_keywords):
                logger.info(f"[SKIPPED] {source_file} (excluded directory)")
                continue
            
            if source_file.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}:
                try:
                    logger.info(f"[source_file] {source_file}")
                    # logger.info(f"[REMAINING] length of remaining_apis: {len(remaining_apis)}")
                    gadget = extract_functions_from_source(source_file, include_dirs)
                    gadgets.extend(gadget)
                    
                except Exception as e:
                    sys.exit(f"AST dump failed: {e}")
                
        
        
        os.makedirs(self.prototype_dir, exist_ok=True)
        file_path = Path(self.prototype_dir) / "chunks.json"
        logger.info(f"[file_path]{file_path}")
        with open(file_path,"w", encoding="utf-8") as f:
            json.dump(gadgets, f)
    
    
    def load_bm25(self):
        """Load or create BM25 index. Loads if existing file exists, otherwise creates new and saves."""
        import pickle
        
        def _log(level, msg):
            if logger:
                getattr(logger, level)(msg)
            else:
                print(f"[{level.upper()}] {msg}")
        
        bm25_path = os.path.join(self.prototype_dir, "bm25_index.pkl")
        chunks_path = os.path.join(self.prototype_dir, "chunks.json")
        
        if os.path.exists(bm25_path):
            _log("info", f"Loading existing BM25 index: {bm25_path}")
            with open(bm25_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.bm25 = saved_data['bm25']
                self.bm25_chunks = saved_data['chunks']
            _log("info", f"BM25 index loaded! ({len(self.bm25_chunks)} chunks)")
            return
        
        if not os.path.exists(chunks_path):
            _log("error", f"chunks.json file does not exist: {chunks_path}")
            raise RuntimeError(f"chunks.json for BM25 creation not found: {chunks_path}")
        
        _log("info", f"Creating new BM25 index from: {chunks_path}")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        tokenized_corpus = []
        for chunk in chunks:
            text_parts = []
            if chunk.get('name'):
                text_parts.append(chunk['name'])
            if chunk.get('return_type'):
                text_parts.append(chunk['return_type'])
            if chunk.get('params'):
                for param in chunk['params']:
                    if param.get('name'):
                        text_parts.append(param['name'])
                    if param.get('type'):
                        text_parts.append(param['type'])
            
            full_text = ' '.join(text_parts)
            tokens = full_text.lower().split()
            tokenized_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_chunks = chunks
        
        os.makedirs(self.prototype_dir, exist_ok=True)
        with open(bm25_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.bm25_chunks}, f)
        _log("info", f"BM25 index created and saved: {bm25_path} ({len(chunks)} chunks)")
    
    def search_bm25(self, query: str, top_n: int = 1):
        """Search with BM25 and return top_n results
        
        Args:
            query: Search query string
            top_n: Number of results to return (default: 1)
            
        Returns:
            list: List of top_n (chunk, score) tuples
        """
        if not hasattr(self, 'bm25') or self.bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call load_bm25() first.")
        
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        results = []
        for idx in top_indices:
            results.append((self.bm25_chunks[idx], scores[idx]))
        
        return results
    
    def load_constraint_bm25(self, lib: str):
        """Load or create Constraint BM25 index
        
        Args:
            lib: Library name
        """
        import pickle
        
        def _log(level, msg):
            if logger:
                getattr(logger, level)(msg)
            else:
                print(f"[{level.upper()}] {msg}")
        
        constraint_dir = os.path.join("rag_store", lib, f"{lib}_constraint_rag")
        os.makedirs(constraint_dir, exist_ok=True)
        
        bm25_path = os.path.join(constraint_dir, "constraint_bm25.pkl")
        chunks_path = os.path.join(constraint_dir, "constraint_chunks.json")
        
        if os.path.exists(bm25_path) and os.path.exists(chunks_path):
            _log("info", f"Loading existing Constraint BM25 index: {bm25_path}")
            with open(bm25_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.constraint_bm25 = saved_data['bm25']
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.constraint_chunks = json.load(f)
            _log("info", f"Constraint BM25 index loaded! ({len(self.constraint_chunks)} chunks)")
        else:
            _log("info", "Creating new Constraint BM25 index")
            self.constraint_bm25 = None
            self.constraint_chunks = []
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
    
    def add_constraint_chunk(self, api_name: str, api_reason: str):
        """Helper: Add API constraint chunk"""
        existing_chunk = None
        for chunk in self.constraint_chunks:
            if chunk.get('api') == api_name:
                existing_chunk = chunk
                break
        
        if existing_chunk:
            reason_count = sum(1 for key in existing_chunk.keys() if key.startswith('reason'))
            existing_chunk[f"reason{reason_count + 1}"] = api_reason
        else:
            chunk = {
                "api": api_name,
                "reason1": api_reason
            }
            self.constraint_chunks.append(chunk)
    
    def add_constraint_to_bm25(self, lib: str, reason: str):
        """Add Constraint to BM25 index
        
        Args:
            lib: Library name
            reason: Reason from LLM (format: "{api1: reason, api2: reason, api3: reason}")
        """
        import pickle
        
        def _log(level, msg):
            if logger:
                getattr(logger, level)(msg)
            else:
                print(f"[{level.upper()}] {msg}")
        
        constraint_dir = os.path.join("rag_store", lib, f"{lib}_constraint_rag")
        os.makedirs(constraint_dir, exist_ok=True)
        
        bm25_path = os.path.join(constraint_dir, "constraint_bm25.pkl")
        chunks_path = os.path.join(constraint_dir, "constraint_chunks.json")
        
        if not hasattr(self, 'constraint_bm25') or self.constraint_bm25 is None:
            self.load_constraint_bm25(lib)
        
        try:
            if isinstance(reason, str):
                import ast
                try:
                    reason_dict = json.loads(reason)
                except:
                    try:
                        reason_dict = ast.literal_eval(reason)
                    except:
                        _log("warning", f"Failed to parse reason, saving as single chunk: {reason[:100]}")
                        reason_dict = {"unknown": reason}
            else:
                reason_dict = reason
        except Exception as e:
            _log("error", f"Error parsing reason: {e}")
            reason_dict = {"unknown": str(reason)}
        
        for api_name, api_reason in reason_dict.items():
            if api_name.lower() in ('reason', 'reason1', 'reason2', 'error', 'unknown', 'note') and isinstance(api_reason, dict):
                for nested_api, nested_reason in api_reason.items():
                    self.add_constraint_chunk(nested_api, str(nested_reason))
                continue
            
            if not api_name or len(api_name) < 3:
                _log("warning", f"Skipping invalid API name: '{api_name}'")
                continue
            
            self.add_constraint_chunk(api_name, str(api_reason))
        
        # Tokenize entire corpus
        tokenized_corpus = []
        for chunk in self.constraint_chunks:
            # Tokenize by combining api, reason1, reason2 etc
            text_parts = []
            if chunk.get('api'):
                text_parts.append(chunk['api'])
            # Process all keys starting with 'reason'
            for key, value in chunk.items():
                if key.startswith('reason'):
                    text_parts.append(str(value))
            
            full_text = ' '.join(text_parts)
            tokens = full_text.lower().split()
            tokenized_corpus.append(tokens)
        
        self.constraint_bm25 = BM25Okapi(tokenized_corpus)
        
        with open(bm25_path, 'wb') as f:
            pickle.dump({'bm25': self.constraint_bm25}, f)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.constraint_chunks, f, indent=2, ensure_ascii=False)
        
        _log("info", f"Constraint BM25 updated: {len(reason_dict)} added, total {len(self.constraint_chunks)}")

    
    def search_constraint_bm25(self, lib: str, api_names: list, top_n: int = 10):
        """Search Constraint BM25 by API names and return results
        
        Args:
            lib: Library name
            api_names: List of API names to search
            top_n: Maximum number of results to return (default: 10)
            
        Returns:
            list: List of (chunk, score) tuples matching API names
        """
        if not hasattr(self, 'constraint_bm25') or self.constraint_bm25 is None:
            self.load_constraint_bm25(lib)
        
        if not self.constraint_chunks:
            return []
        
        api_names_lower = [api.lower() for api in api_names]
        
        matched_results = []
        for chunk in self.constraint_chunks:
            chunk_api = chunk.get('api', '').lower()
            if chunk_api in api_names_lower:
                matched_results.append((chunk, 100.0))
        
        if matched_results:
            return matched_results[:top_n]
        
        query = ' '.join(api_names)
        tokens = query.lower().split()
        scores = self.constraint_bm25.get_scores(tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.constraint_chunks[idx], scores[idx]))
        
        return results

        
    def build_library_rag(self):
        self.collect_files(self.lib_dir, self.lib_docs_dir, self.file_exts)
        
        if not os.path.exists(self.index_dir):
            logger.info("Indexing documents...")
            documents = SimpleDirectoryReader(self.lib_docs_dir).load_data()
            index = GPTVectorStoreIndex.from_documents(documents, embed_model=self.embed_model)
            index.storage_context.persist(self.index_dir)            
            logger.info("Index creation complete!")
        else:
            logger.info(f"Existing index({self.index_dir}) found. Delete folder and re-run if you changed embedding method.")

    def build_api_prototype(self, api_list: List, include_dirs:List):
        logger.info("Indexing AST...")
        self.extract_ast_rag(api_list=api_list,include_dirs=include_dirs)
        logger.info("AST generation complete!")
    
    def load_index_and_engine(self):
        """Load existing index and initialize query engine"""
        if os.path.exists(self.index_dir):
            Settings.embed_model = self.embed_model
            storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
            self.index = load_index_from_storage(storage_context)

            # self.query_engine = self.index.as_query_engine()
            with open(os.path.join(os.path.dirname(__file__), "prompts", "categorize_api_template.txt"), "r") as f:
                prompt = f.read()
                template = RichPromptTemplate(prompt)
                self.query_engine["categorize"] = self.index.as_query_engine(text_qa_template=template, similarity_top_k=3)
            f.close()
            with open(os.path.join(os.path.dirname(__file__), "prompts", "make_harness_template.txt"), "r") as f:
                prompt = f.read()
                template = RichPromptTemplate(prompt)
                self.query_engine["make"] = self.index.as_query_engine(text_qa_template=template, similarity_top_k=3)
            f.close()
            with open(os.path.join(os.path.dirname(__file__), "prompts", "fix_harness_template.txt"), "r") as f:
                prompt = f.read()
                template = RichPromptTemplate(prompt)
                self.query_engine["fix"] = self.index.as_query_engine(text_qa_template=template, similarity_top_k=3)
            f.close()
            with open(os.path.join(os.path.dirname(__file__), "prompts", "make_api_flow_template.txt"), "r") as f:
                prompt = f.read()
                template = RichPromptTemplate(prompt)
                self.query_engine["flow"] = self.index.as_query_engine(text_qa_template=template, similarity_top_k=3)
            f.close()
            with open(os.path.join(os.path.dirname(__file__), "prompts", "judgement_template.txt"), "r") as f:
                prompt = f.read()
                template = RichPromptTemplate(prompt)
                self.query_engine["judgement"] = self.index.as_query_engine(text_qa_template=template, similarity_top_k=3)
            f.close()
        else:
            logger.error(f"RAG index directory does not exist: {self.index_dir}")
            raise RuntimeError(f"RAG index not built: {self.index_dir}")

    def _get_token_delta(self) -> tuple:
        """Read per-call token counts, reset counter, and invoke token_callback if registered."""
        in_tok  = self.token_counter.prompt_llm_token_count
        out_tok = self.token_counter.completion_llm_token_count
        self.token_counter.reset_counts()
        if self.token_callback:
            self.token_callback(in_tok, out_tok)
        return in_tok, out_tok


    async def query_categorize(self, lib: str, api_list: list, unused_apis: list = None):
        query_text = (
            f"TASK=CATEGORIZE_APIS\n"
            f"lib={lib}\n"
            f"apis={api_list}"
        )
        
        if unused_apis and len(unused_apis) > 0:
            apis_to_show = unused_apis[:100]
            query_text += f"\n\nUNUSED APIs (consider including some of these if they fit naturally):\n{apis_to_show}"
            logger.info(f"[RAG] Unused API feedback added ({len(apis_to_show)} out of {len(unused_apis)} APIs)")
        
        self.llama_debug.flush_event_logs()
        
        self.prompt_logger.current_task = "CATEGORIZE"
        self.prompt_logger.current_lib = lib
        self.prompt_logger.current_category = "N/A"
        
        start_time = time.time()
        try:
            response = await asyncio.to_thread(self.query_engine["categorize"].query, query_text)
            total_time = time.time() - start_time
            
            llm_events = self.llama_debug.get_event_pairs(CBEventType.LLM)
            llm_time = 0
            if llm_events:
                llm_time = (datetime.datetime.strptime(llm_events[-1][1].time, "%m/%d/%Y, %H:%M:%S.%f") - datetime.datetime.strptime(llm_events[-1][0].time, "%m/%d/%Y, %H:%M:%S.%f")).total_seconds()
            rag_time = total_time - llm_time
            
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("CATEGORIZE", lib, "N/A", len(query_text), total_time, rag_time, llm_time, in_tok, out_tok)
            logger.info(f"=== [PERF] CATEGORIZE completed (file saved) ===")
            logger.info(f" - Total: {total_time:.2f}s (RAG: {rag_time:.2f}s, LLM: {llm_time:.2f}s) | IN_TOK: {in_tok} | OUT_TOK: {out_tok}")
            
            return response
        except Exception as e:
            total_time = time.time() - start_time
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("CATEGORIZE_FAIL", lib, "N/A", len(query_text), total_time, 0, total_time, in_tok, out_tok)
            logger.error(f"[RAG] Categorize query failed: {e}")
            raise e

    async def query_make(self, lib: str, category: str, api_flow: list):
        # Get compile command from build_config.yaml if available
        compile_cmd = ""
        try:
            config = load_build_config()
            config_key = f"{lib}_fuzz"
            if config_key in config and "build_harness" in config[config_key]:
                build_harness = config[config_key]["build_harness"]
                if isinstance(build_harness, list):
                    compile_cmd = "\n".join(build_harness)
                else:
                    compile_cmd = str(build_harness)
        except Exception as e:
            logger.warning(f"Failed to load compile command for make query: {e}")

        query_text = (
            f"TASK=MAKE_HARNESS\n"
            f"lib={lib}\n"
            f"category={category}\n"
            f"api_flow={api_flow}"
        )
        if compile_cmd:
            query_text += f"\ncompile_command:\n{compile_cmd}"

        self.llama_debug.flush_event_logs()
        
        self.prompt_logger.current_task = "MAKE"
        self.prompt_logger.current_lib = lib
        self.prompt_logger.current_category = category
        
        start_time = time.time()
        try:
            response = await asyncio.to_thread(self.query_engine["make"].query, query_text)
            total_time = time.time() - start_time
            
            llm_events = self.llama_debug.get_event_pairs(CBEventType.LLM)
            llm_time = 0
            if llm_events:
                llm_time = (datetime.datetime.strptime(llm_events[-1][1].time, "%m/%d/%Y, %H:%M:%S.%f") - datetime.datetime.strptime(llm_events[-1][0].time, "%m/%d/%Y, %H:%M:%S.%f")).total_seconds()
            rag_time = total_time - llm_time
            
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("MAKE", lib, category, len(query_text), total_time, rag_time, llm_time, in_tok, out_tok)
            logger.info(f"=== [PERF] MAKE_HARNESS completed (file saved) ===")
            logger.info(f" - Total: {total_time:.2f}s (RAG: {rag_time:.2f}s, LLM: {llm_time:.2f}s) | IN_TOK: {in_tok} | OUT_TOK: {out_tok}")
            
            return response
        except Exception as e:
            total_time = time.time() - start_time
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("MAKE_FAIL", lib, category, len(query_text), total_time, 0, total_time, in_tok, out_tok)
            logger.error(f"[RAG] Make query failed (Category: {category}): {e}")
            raise e


    async def query_fix(self, lib: str, category: str, api_list: list, original_harness: str, build_error: str):
        # Get compile command from build_config.yaml if available
        compile_cmd = ""
        try:
            config = load_build_config()
            config_key = f"{lib}_fuzz"
            if config_key in config and "build_harness" in config[config_key]:
                build_harness = config[config_key]["build_harness"]
                if isinstance(build_harness, list):
                    compile_cmd = "\n".join(build_harness)
                else:
                    compile_cmd = str(build_harness)
        except Exception as e:
            logger.warning(f"Failed to load compile command for fix query: {e}")

        api_list_str = ", ".join(api_list)
        harness_snippet = (original_harness or "")
        build_err_short = (build_error or "")
        query_text = (
            "TASK=FIX_HARNESS\n"
            f"lib={lib}\n"
            f"category={category}\n"
            "original_code:\n" + harness_snippet + "\n"
            "build_error:\n" + build_err_short
        )
        if compile_cmd:
            query_text += f"\ncompile_command:\n{compile_cmd}"

        self.llama_debug.flush_event_logs()
        
        self.prompt_logger.current_task = "FIX"
        self.prompt_logger.current_lib = lib
        self.prompt_logger.current_category = category
        
        start_time = time.time()
        try:
            response = await asyncio.to_thread(self.query_engine["fix"].query, query_text)
            total_time = time.time() - start_time
            
            llm_events = self.llama_debug.get_event_pairs(CBEventType.LLM)
            llm_time = 0
            if llm_events:
                llm_time = (datetime.datetime.strptime(llm_events[-1][1].time, "%m/%d/%Y, %H:%M:%S.%f") - datetime.datetime.strptime(llm_events[-1][0].time, "%m/%d/%Y, %H:%M:%S.%f")).total_seconds()
            rag_time = total_time - llm_time
            
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("FIX", lib, category, len(query_text), total_time, rag_time, llm_time, in_tok, out_tok)
            logger.info(f"=== [PERF] FIX_HARNESS completed (file saved) ===")
            logger.info(f" - Total: {total_time:.2f}s (RAG: {rag_time:.2f}s, LLM: {llm_time:.2f}s) | IN_TOK: {in_tok} | OUT_TOK: {out_tok}")
            
            return response
        except Exception as e:
            total_time = time.time() - start_time
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("FIX_FAIL", lib, category, len(query_text), total_time, 0, total_time, in_tok, out_tok)
            logger.error(f"[RAG] Fix query failed (Category: {category}): {e}")
            raise e


    async def query_flow(self, lib: str, category: str, api_list: list, all_apis: list, unused_apis: list = None):
        query_text = (
            f"TASK=MAKE_API_FLOW\n"
            f"lib={lib}\n"
            f"category={category}\n"
            f"apis={api_list}\n"
            f"all_apis={all_apis}"
        )
        
        if unused_apis and len(unused_apis) > 0:
            category_unused = [api for api in api_list if api in unused_apis]
            if category_unused:
                query_text += f"\n\nUNUSED APIs in this category (consider including if they fit the flow naturally):\n{category_unused}"
        
        self.llama_debug.flush_event_logs()
        
        self.prompt_logger.current_task = "FLOW"
        self.prompt_logger.current_lib = lib
        self.prompt_logger.current_category = category
        
        start_time = time.time()
        try:
            response = await asyncio.to_thread(self.query_engine["flow"].query, query_text)
            total_time = time.time() - start_time
            
            llm_events = self.llama_debug.get_event_pairs(CBEventType.LLM)
            llm_time = 0
            if llm_events:
                llm_time = (datetime.datetime.strptime(llm_events[-1][1].time, "%m/%d/%Y, %H:%M:%S.%f") - datetime.datetime.strptime(llm_events[-1][0].time, "%m/%d/%Y, %H:%M:%S.%f")).total_seconds()
            rag_time = total_time - llm_time
            
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("FLOW", lib, category, len(query_text), total_time, rag_time, llm_time, in_tok, out_tok)
            logger.info(f"=== [PERF] MAKE_API_FLOW completed (file saved) ===")
            logger.info(f" - Total: {total_time:.2f}s (RAG: {rag_time:.2f}s, LLM: {llm_time:.2f}s) | IN_TOK: {in_tok} | OUT_TOK: {out_tok}")
            
            return response
        except Exception as e:
            total_time = time.time() - start_time
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("FLOW_FAIL", lib, category, len(query_text), total_time, 0, total_time, in_tok, out_tok)
            logger.error(f"[RAG] Flow query failed (Category: {category}): {e}")
            raise e

    
    async def query_judgement(
        self, 
        lib: str,
        category: str,
        harness_code: str,
        fuzz_stats: dict,
        crash_outputs: list
    ) -> Tuple[bool, str]:
        """
        LLM analyzes harness and crash info to determine whether to proceed with long fuzzing.
        Uses RAG query engine with library documentation context.
        
        Returns:
            (should_continue, reason): Whether to proceed with long fuzzing, reasoning
        """
        crash_outputs_str = "\n".join(crash_outputs) if crash_outputs else "No crashes found."
        fuzz_stats_str = json.dumps(fuzz_stats, indent=2) if fuzz_stats else "No statistics available."
        
        # Build query text with all context
        query_text = (
            f"TASK=JUDGEMENT\n"
            f"lib={lib}\n"
            f"category={category}\n"
            f"harness_code={harness_code or ''}\n"
            f"crash_outputs={crash_outputs_str}\n"
            f"fuzz_stats={fuzz_stats_str}"
        )
        
        self.llama_debug.flush_event_logs()
        
        self.prompt_logger.current_task = "JUDGEMENT"
        self.prompt_logger.current_lib = lib
        self.prompt_logger.current_category = category
        
        start_time = time.time()
        logger.info(f"[RAG] LLM Judgement start - {category}")
        
        try:
            # Temporarily set temperature to 0 for deterministic judgement
            original_temp = self.llm.temperature
            self.llm.temperature = 0.0
            
            response = await asyncio.to_thread(self.query_engine["judgement"].query, query_text)
            total_time = time.time() - start_time
            response_text = str(response)
            
            self.llm.temperature = original_temp
            
            llm_events = self.llama_debug.get_event_pairs(CBEventType.LLM)
            llm_time = 0
            if llm_events:
                llm_time = (datetime.datetime.strptime(llm_events[-1][1].time, "%m/%d/%Y, %H:%M:%S.%f") - datetime.datetime.strptime(llm_events[-1][0].time, "%m/%d/%Y, %H:%M:%S.%f")).total_seconds()
            rag_time = total_time - llm_time
            
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("JUDGEMENT", lib, category, len(query_text), total_time, rag_time, llm_time, in_tok, out_tok)
            logger.info(f"=== [PERF] JUDGEMENT completed (file saved) ===")
            logger.info(f" - Total: {total_time:.2f}s (RAG: {rag_time:.2f}s, LLM: {llm_time:.2f}s) | IN_TOK: {in_tok} | OUT_TOK: {out_tok}")
            
            logger.info(f"[RAG] LLM Judgement complete - response length: {len(response_text)}")
            logger.debug(f"[RAG] LLM Judgement raw response:\n{response_text}")
            
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.info(f"[RAG] JSON block extraction successful")
                else:
                    brace_match = re.search(r'\{[^{}]*"should_continue"[^{}]*\}', response_text, re.DOTALL)
                    if brace_match:
                        json_str = brace_match.group(0)
                        logger.info(f"[RAG] JSON object direct extraction successful")
                    else:
                        logger.warning(f"[RAG] No JSON format found, trying keyword-based judgment")
                        response_lower = response_text.lower()
                        if "reject" in response_lower or "should_continue\": false" in response_lower or "should_continue\":false" in response_lower:
                            return False, f"{response_text}"
                        else:
                            return True, f"{response_text}"
                
                judgement = json.loads(json_str)
                should_continue = judgement.get("should_continue", False)
                reason = judgement.get("reason", "No reason provided")
                logger.info(f"[RAG] JSON parsing successful: should_continue={should_continue}")
                return should_continue, reason
            except json.JSONDecodeError as e:
                logger.warning(f"[RAG] JSON parsing failed: {e}")
                logger.warning(f"[RAG] Failed JSON string: {json_str[:200] if 'json_str' in dir() else 'N/A'}")
                return True, f"JSON parsing failed - proceeding with default: {response_text[:100]}"
                
        except Exception as e:
            total_time = time.time() - start_time
            in_tok, out_tok = self._get_token_delta()
            save_performance_detailed_log("JUDGEMENT_FAIL", lib, category, len(query_text), total_time, 0, total_time, in_tok, out_tok)
            logger.error(f"[RAG] LLM Judgement error: {e}")
            return True, f"LLM call failed - proceeding with default: {str(e)}"

def load_build_config(config_path: str = "build_config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config file: {e}")
        return {}

def install_dependencies(dependencies: list) -> Tuple[bool, str]:
    """Install dependency packages in the current container only."""
    if not dependencies:
        return True, "No dependencies"

    # Check which packages are not yet installed in current container using dpkg -s
    missing = []
    for dep in dependencies:
        res = subprocess.run(["dpkg", "-s", dep], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            missing.append(dep)

    if not missing:
        logger.info("[dependency] All dependencies already installed in container.")
        return True, "All dependencies already installed.\n"

    log = ""
    logger.info("[dependency] apt-get update")
    process = subprocess.Popen(["apt-get", "update"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        log += line
    process.wait()
    if process.returncode != 0:
        return False, f"Failed to update package list\n{log}"

    for dep in missing:
        cmd = ["apt-get", "install", "-y", dep]
        logger.info(f"[dependency] {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log += line
        process.wait()
        if process.returncode != 0:
            return False, f"Failed to install dependency: {dep}\n{log}"

    return True, log


def sync_image_dependencies(config_path: str = "build_config.yaml"):
    """Collect all dependencies from build_config.yaml and ensure the afk image has them.
    Call once at agent startup."""
    config = load_build_config(config_path)
    if not config:
        return

    all_deps = set()
    for lib_config in config.values():
        for dep in lib_config.get("dependencies", []):
            all_deps.add(dep)

    if not all_deps:
        return

    # Check which deps the image is missing
    try:
        check_script = " ".join([
            f"dpkg -s {dep} >/dev/null 2>&1 || echo {dep};"
            for dep in sorted(all_deps)
        ])
        cmd = ["docker", "run", "--rm", "afk:latest", "sh", "-c", check_script]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=60
        )
        missing = [p for p in result.stdout.splitlines() if p.strip()]
    except Exception as e:
        logger.warning(f"[dependency] Failed to check afk image: {e}")
        missing = list(all_deps)

    if not missing:
        logger.info("[dependency] afk image already has all dependencies.")
        return

    logger.info(f"[dependency] afk image missing {len(missing)} packages: {missing}")
    try:
        pkgs = " ".join(missing)
        tmpdir = os.path.join("/tmp", f"afk_dep_{os.getpid()}")
        os.makedirs(tmpdir, exist_ok=True)
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
            f.write("FROM afk:latest\n")
            f.write(f"RUN apt-get update && apt-get install -y {pkgs} && rm -rf /var/lib/apt/lists/*\n")

        process = subprocess.Popen(
            ["docker", "build", "--network=host", "-t", "afk:latest", tmpdir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            logger.info(f"[docker build] {line.rstrip()}")
        process.wait()
        if process.returncode == 0:
            logger.info(f"[dependency] Rebuilt afk:latest with {len(missing)} new packages")
        else:
            logger.error(f"[dependency] docker build failed with returncode {process.returncode}")

        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        logger.error(f"[dependency] Failed to rebuild afk image: {e}")

def get_library_paths(library_name: str) -> Tuple[str, str]:
    """
    For library_name like library_api, library_fuzz, etc.,
    split into (library_base, purpose) and return
    library/{library_base}/{library_name} path.
    """
    if '_' in library_name:
        base, purpose = library_name.split('_', 1)
    else:
        base, purpose = library_name, 'default'
    base_path = os.path.join('library', base)
    full_path = os.path.join(base_path, library_name)
    return base_path, full_path

def clone_and_build_library(library_name: str, config_path: str = "build_config.yaml") -> bool:
    """
    Clone and build the library according to the YAML configuration.
    Use directory structure: library/{library_base}/{library_name} for each purpose.
    """
    logger.info(f"[clone_and_build_library] clone and build {library_name}")
    config = load_build_config(config_path)
    if library_name not in config:
        return False
    lib_config = config[library_name]
    env_vars = lib_config.get("env") or {}
    build_steps = lib_config.get("build_library") or []
    for k, v in env_vars.items():
        os.environ[k] = v
    repo_url = lib_config.get("repo_url")
    branch = lib_config.get("branch", "master")

    _, build_dir = get_library_paths(library_name)
    dependencies = lib_config.get("dependencies") or []
    build_type = lib_config.get("build_type", "autotools")
    original_dir = os.getcwd()
    try:
        if os.path.exists(build_dir):
            logger.info(f"Directory already exists: {build_dir}")
            return True

        if dependencies:
            logger.info("=== Start installing dependencies ===")
            dep_ok, dep_log = install_dependencies(dependencies)
            if not dep_ok:
                return False

        # Determine if this is a commit hash or branch (40 hex chars = commit hash)
        is_commit_hash = len(branch) == 40 and all(c in '0123456789abcdef' for c in branch.lower())
        
        if is_commit_hash:
            # For commit hash: clone then checkout
            clone_cmd = ["git", "clone", repo_url, build_dir]
            logger.info(f"[clone] {' '.join(clone_cmd)}")
            process = subprocess.Popen(clone_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                logger.info(line.rstrip())
            process.wait()
            if process.returncode != 0:
                return False
            
            # checkout specific commit
            checkout_cmd = ["git", "-C", build_dir, "checkout", branch]
            logger.info(f"[checkout] {' '.join(checkout_cmd)}")
            process = subprocess.Popen(checkout_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                logger.info(line.rstrip())
            process.wait()
            if process.returncode != 0:
                return False
        else:
            # For branch/tag: use --branch option
            clone_cmd = ["git", "clone", "--branch", branch, repo_url, build_dir]
            logger.info(f"[clone] {' '.join(clone_cmd)}")
            process = subprocess.Popen(clone_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                logger.info(line.rstrip())
            process.wait()
            if process.returncode != 0:
                return False

        os.chdir(build_dir)
        for step in build_steps:
            logger.info(f"[build] {step}")
            if step.startswith("cd "):
                target_dir = step[3:].strip()
                os.chdir(target_dir)
                logger.info(f"cd {target_dir}")
                continue
            process = subprocess.Popen(step, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                logger.info(line.rstrip())
            process.wait()
            if process.returncode != 0:
                return False
        logger.info(f"=== {library_name} build success ===")
        return True
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return False
    finally:
        os.chdir(original_dir)

def build_check(library_name: str):
    _,build_dir = get_library_paths(library_name)
    if os.path.exists(build_dir):
        return 0
    else:
        return 1

async def build_fuzzer(
    library: str,
    category: str,
    config_path: str = "build_config.yaml",
    purpose: str = "",
    run_index: int = 0,
    gen_try: int = 0,
    fix_try: int = 0,
) -> Tuple[bool, str, str]:
    """
    Execute build_harness commands in category directory, substituting {output}, {main_cc}, {harness_dir}, {harness_path}, {lib}, {lib_base}.
    :param library: Library name (e.g., libpng)
    :param category: Category name (e.g., Cleanup)
    :param config_path: build_config.yaml path
    :param purpose: Build purpose (e.g., 'fuzz', 'cov')
    :param output: Build result binary filename
    :return: (success, build_log, binary_path)
    """
    safe_category = re.sub(r'[^A-Za-z0-9_]', '_', category)
    harness_category_dir = os.path.abspath(os.path.join("library", library, f"harness_{run_index:03d}", safe_category))
    
    gen_dir = f"gen{gen_try}_fix{fix_try}"
    gen_path = os.path.join(harness_category_dir, gen_dir)
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    
    # Binary name differentiation based on purpose (saved inside gen directory)
    if purpose == "cov":
        output_path = os.path.join(gen_path, f"{library}_{safe_category}_cov")
    elif purpose == "api":
        output_path = os.path.join(gen_path, f"{library}_{safe_category}_reproduce")
    else:
        output_path = os.path.join(gen_path, f"{library}_{safe_category}_fuzzer")
    
    category_dir = gen_path
    logger.info(f"[build_fuzzer] {category_dir} build start (output: {output_path})")

    config = load_build_config(config_path)
    config_key = f"{library}_{purpose}"
    build_log = ""
    success = True

    if config_key not in config or "build_harness" not in config[config_key]:
        logger.error(f"[build_fuzzer] build_harness not found: {config_key}")
        build_log += "[build_harness][ERROR] build_harness not found in build_config.yaml.\n"
        return False, build_log, ""

    env_vars = config[config_key].get("env") or {}
    logger.info(f"[build_fuzzer] Applying env vars: {env_vars}")
    for k, v in env_vars.items():
        os.environ[k] = v

    build_harness_steps = config[config_key]["build_harness"]

    main_cc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.cc"))
    harness_dir = gen_path
    harness_path = os.path.abspath(os.path.join(gen_path, f"{safe_category}_harness.c"))
    
    lib_base = os.path.abspath(os.path.join("library", library, f"{library}_{purpose}"))

    for step in build_harness_steps:
        cmd = (
            step.replace("{output}", output_path)
                .replace("{main_cc}", main_cc_path)
                .replace("{harness_dir}", harness_dir)
                .replace("{harness_path}", harness_path)
                .replace("{lib_base}", lib_base)
        )
        logger.info(f"[build_fuzzer] Executing command: {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=category_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        returncode = proc.returncode
        stdout_str = stdout.decode('utf-8') if stdout else ""
        stderr_str = stderr.decode('utf-8') if stderr else ""
        build_log += f"[build_harness] {cmd}\n"
        build_log += stdout_str + "\n" + stderr_str
        if returncode != 0:
            logger.error(f"[build_fuzzer] Build failed: {cmd}")
            success = False
            break
    if success:
        logger.info(f"[build_fuzzer] Build success: {output_path}")
    else:
        logger.error(f"[build_fuzzer] Build failed: {category_dir}")

    binary_path = output_path if success else ""
    return success, build_log, binary_path

def extract_lib_symbols(lib_path: str) -> list:
    """
    Extract exported function (API) list from the library using nm.
    """
    logger.info(f"Extracting symbols with nm: {lib_path}")
    try:
        # Use string command instead of list for proper pipe handling
        cmd = f"nm --no-demangle --defined-only -g {lib_path} | awk '$2==\"T\" {{print $3}}' | grep -v '^_' | sort -u"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        
        api_list = result.stdout.strip().splitlines()
        if api_list:
            logger.info(f"Extracted {len(api_list)} APIs (nm)")
            return api_list
        else:
            logger.info("C-style symbol extraction yielded no symbols. Switching to C++ symbol extraction mode.")
    except Exception as e:
        logger.error(f"Failed to extract APIs: {e}")
        logger.error(f"Command return code: {getattr(e, 'returncode', 'N/A')}")
        logger.error(f"Command stdout: {getattr(e, 'stdout', 'N/A')}")
        logger.error(f"Command stderr: {getattr(e, 'stderr', 'N/A')}")
        logger.info("Attempting C++ symbol extraction using llvm-nm demangler")

    try:
        fallback_cmd = (
            f"llvm-nm -D --defined-only -g -j --demangle {lib_path} "
            f"| sed 's/@.*//' "
            f"| grep -Ev -i '(^$|__cxa_atexit$|__dso_handle$|__gmon_start__$|_init$|_fini$)' "
            f"| grep -Ev -i '(vtable for|typeinfo( name)? for|RTTI|non-virtual thunk|virtual thunk|construction vtable|guard variable)' "
            f"| grep -Ev '(^_ZTI|^_ZTS|^_ZTV|_ZThn|_ZTv|_ZTT)' "
            f"| grep -Ev '^(std|__gnu_cxx|__cxxabiv1|abi)::' "
            f"| grep -Ev '^std::__' "
            f"| grep -Ev '<.*>' "
            f"| sort -u"
        )
        result_fb = subprocess.run(fallback_cmd, shell=True, capture_output=True, text=True, check=True)
        fb_list = [s for s in result_fb.stdout.strip().splitlines() if s]
        logger.info(f"Extracted {len(fb_list)} APIs (C++ Symbol Extractor)")
        return fb_list
    except Exception as e2:
        logger.error(f"C++ symbol extraction also failed: {e2}")
        logger.error(f"stdout: {getattr(e2, 'stdout', 'N/A')}")
        logger.error(f"stderr: {getattr(e2, 'stderr', 'N/A')}")
        return []

def generate_ast_files(lib: str) -> Tuple[bool, str]:
    """
    Generate AST (.ast) files from the source files of the built library.
    Store them in ast/ folder.
    """
    _, lib_dir = get_library_paths(lib)
    logger.info(f"=== Start generating AST: {lib} ===")
    source_files = []
    for root, dirs, files in os.walk(lib_dir):
        for file in files:
            if file.endswith(('.c', '.cpp', '.cc')):
                source_files.append(os.path.join(root, file))
    if not source_files:
        logger.warning(f"No source files found: {lib_dir}")
        return False, "No source files found"
    logger.info(f"Found {len(source_files)} source files")

    library_base = lib.replace('_api', '').replace('_fuzz', '')
    ast_dir = os.path.join('library', library_base, 'ast')
    os.makedirs(ast_dir, exist_ok=True)
    success_count = 0
    error_count = 0
    log = f"=== AST generation log ===\n"
    base_path, _ = get_library_paths(lib)
    include_dirs = [lib_dir, base_path]
    common_include_dirs = ["include", "src", "lib", "headers", "inc"]
    for include_dir in common_include_dirs:
        full_path = os.path.join(lib_dir, include_dir)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            include_dirs.append(full_path)
    
    for source_file in source_files:
        try:
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            ast_file = os.path.join(ast_dir, f"{base_name}.ast")
            
            include_args = []
            for include_dir in include_dirs:
                include_args.extend(["-I", include_dir])
            
            ast_cmd = [
                "clang", "-Xclang", "-ast-dump", "-fsyntax-only"
            ] + include_args + [source_file]
            logger.info(f"Generating AST: {source_file} -> {ast_file}")
            result = subprocess.run(ast_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                with open(ast_file, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                log += f"AST success: {base_name}.ast\n"
                success_count += 1
            else:
                log += f"AST failed: {base_name} - {result.stderr}\n"
                error_count += 1
        except Exception as e:
            logger.error(f"Error processing file: {source_file} - {e}")
            log += f"Error: {os.path.basename(source_file)} - {e}\n"
            error_count += 1

    log += f"\n=== AST generation complete ===\n"
    log += f"Success: {success_count}, Failed: {error_count}\n"
    log += f"Output location: {ast_dir}\n"
    logger.info(f"AST generation complete: Success {success_count}, Failed {error_count}")
    logger.info(f"Output location: {ast_dir}")
    return success_count > 0, log

def prepare_library_docs(
    docs_path: str = "library_docs",
    library_base: str = "library_base",
    exts = [".h", ".c", ".cc", ".cpp", ".txt", ".md"]
):
    """
    Create library_docs folder if it doesn't exist, and copy .h/.c/.cc/.cpp/.txt/.md files from all subdirs of library_base.
    Log existence after directory creation, log error on failure.
    """
    docs_path = os.path.abspath(docs_path)
    library_base = os.path.abspath(library_base)
    logger.info(f"Attempting to create library_docs: {docs_path}")
    try:
        os.makedirs(docs_path, exist_ok=True)
        logger.info(f"library_docs directory created: {os.path.exists(docs_path)}")
    except Exception as e:
        logger.error(f"Failed to create library_docs directory: {e}")
        raise
    copied = 0
    for root, _, files in os.walk(library_base):
        for file in files:
            if any(file.endswith(ext) for ext in exts):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(docs_path, file)
                if not os.path.exists(dst_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied += 1
                        logger.info(f"Copied: {src_path} -> {dst_path}")
                    except Exception as e:
                        logger.warning(f"File copy failed: {src_path} -> {dst_path} ({e})")
    if not os.listdir(docs_path):
        raise RuntimeError(f"No files found in {docs_path}.")
    logger.info(f"Files copied to library_docs: {copied}")


async def run_docker_fuzzing(library_name: str, fuzzer_name: str, duration_seconds: int = 0, run_index: int = 0, cpuset_cpus: Optional[str] = None, gen_try: int = 0, fix_try: int = 0) -> bool:
    """
    Run fuzzing in Docker container using AFL++.
    """
    container_name = f"{fuzzer_name}_container"
    logger.info(f"=== Docker fuzzing start for {library_name} (fuzzer: {fuzzer_name}, container: {container_name}, duration: {duration_seconds} seconds) ===")
    
    try:
        client = docker.from_env()
        try:
            container = client.containers.get(container_name)
            logger.info(f"Found existing container {container_name}, force cleaning...")
            
            if container.status == 'running':
                try:
                    container.stop(timeout=60)
                    logger.info(f"Existing container {container_name} stopped")
                except Exception as stop_error:
                    logger.warning(f"Normal stop failed, attempting force stop: {stop_error}")
                    try:
                        subprocess.run(["docker", "kill", container_name], 
                                     capture_output=True, timeout=60, text=True)
                    except:
                        pass
            
            container.remove(force=True)
            logger.info(f"Existing container {container_name} cleaned")
            
        except docker.errors.NotFound:
            logger.info(f"No existing container {container_name} (normal)")
        except Exception as e:
            logger.warning(f"Failed to clean existing container, force cleaning via subprocess: {e}")
            try:
                subprocess.run(["docker", "rm", "-f", container_name], 
                             capture_output=True, timeout=60, text=True)
                logger.info(f"Force cleaned existing container via subprocess")
            except Exception as force_error:
                logger.warning(f"Force cleaning also failed: {force_error}")
        
        logger.info(f"Running docker container...")
        category_name = fuzzer_name.replace(f'{library_name}_', '').replace('_fuzzer', '')
        
        harness_root = f"library/{library_name}/harness_{run_index:03d}"
        category_root = f"{harness_root}/{category_name}"
        gen_dir = f"gen{gen_try}_fix{fix_try}"
        gen_root = f"{category_root}/{gen_dir}"
        out_dir = f"{gen_root}/out"
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Fuzzing output directory created: {out_dir} (attempt: {gen_dir})")
        
        shared_in_dir = f"{gen_root}/in"
        
        corpus_in_dir = f"corpus/{library_name}/in"
        if os.path.exists(corpus_in_dir):
            try:
                shutil.copytree(corpus_in_dir, shared_in_dir, dirs_exist_ok=True)
                logger.info(f"Copied corpus/{library_name}/in directory to {shared_in_dir}.")
            except Exception as e:
                logger.warning(f"Seed directory copy failed: {corpus_in_dir} -> {shared_in_dir}: {e}")
        else:
            logger.warning(f"corpus/{library_name}/in directory does not exist.")

        # Mount entire HOST_BASE_PATH as read-only,
        # Mount only the out directory that needs write access as read-write.
        #   - Host: {HOST_BASE_PATH}                  -> Container: /afk                    (ro)
        #   - Host: {HOST_BASE_PATH}/{out_dir}        -> Container: /afk/{out_dir}          (rw)
        host_out_dir = os.path.join(HOST_BASE_PATH, out_dir)
        os.makedirs(host_out_dir, exist_ok=True)
        host_library_dir = os.path.join(HOST_BASE_PATH, "library")
        container = client.containers.run(
            image="afk",
            name=container_name,
            privileged=True,
            detach=True,
            tty=True,
            stdin_open=True,
            network_mode="none",
            volumes=[
                f"{HOST_BASE_PATH}:/afk:ro",
                f"{host_library_dir}:/afk/library:rw"
            ],
            cpuset_cpus=cpuset_cpus
        )
        logger.info(f"Docker container started successfully: {container.id}")

        # 3. Set core_pattern inside container and run afl-fuzz
        exit_code, output = container.exec_run("/bin/sh -c 'echo core | tee /proc/sys/kernel/core_pattern'")
        logger.info(f"core_pattern setup result: {output.decode() if output else ''}")

        in_dir_container = (
            f"/afk/library/{library_name}/harness_{run_index:03d}/"
            f"{category_name}/{gen_dir}/in"
        )
        out_dir_container = (
            f"/afk/library/{library_name}/harness_{run_index:03d}/"
            f"{category_name}/{gen_dir}/out"
        )
        target_bin = (
            f"/afk/library/{library_name}/harness_{run_index:03d}/"
            f"{category_name}/{gen_dir}/{fuzzer_name}"
        )

        afl_fuzz_cmd = (
            f"AFL_SKIP_CPUFREQ=1 AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 AFL_SKIP_CRASHES=1 AFL_IGNORE_SEED_PROBLEMS=1 AFL_IGNORE_PROBLEMS=1 AFL_NO_AFFINITY=1 "
            f"timeout -s SIGKILL {duration_seconds} "
            f"afl-fuzz -i {in_dir_container} "
            f"-o {out_dir_container} "
            f"-- {target_bin}"
        )
        logger.info(f"Running AFL++ fuzzer... (duration: {duration_seconds} seconds)")

        container.exec_run(cmd=["/bin/sh", "-lc", afl_fuzz_cmd], detach=True)
        logger.info("AFL++ fuzzer started in background (exec_run)")
        await asyncio.sleep(2)
        logger.info(f"AFL++ fuzzer is running in background. (duration: {duration_seconds} seconds)")
        logger.info(f"=== Docker container and fuzzer setup completed for {library_name} ===")
        return True
        
    except Exception as e:
        logger.error(f"Exception during docker fuzzing: {e}")
        return False


async def measure_coverage(library_name: str, fuzzer_name: str, run_index: int = 0, gen_try: int = 0, fix_try: int = 0) -> Optional[float]:
    """
    Measure code coverage from fuzzing results.
    """
    container_name = f"{fuzzer_name}_container"
    logger.info(f"=== Coverage measurement start for {library_name} (fuzzer: {fuzzer_name}, container: {container_name}) ===")
    
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        logger.info("Starting coverage measurement with llvm-cov...")
        logger.info("Step 1: Copying fuzzing results to corpus directory...")
        
        category_name = fuzzer_name.replace(f'{library_name}_', '').replace('_fuzzer', '')
        harness_root = f"library/{library_name}/harness_{run_index:03d}"
        category_root = f"{harness_root}/{category_name}"
        
        gen_dir = f"gen{gen_try}_fix{fix_try}"
        active_dir = os.path.join(category_root, gen_dir)
        logger.info(f"Coverage measurement directory: {active_dir}")
        
        corpus_dir = f"{active_dir}/corpus"
        os.makedirs(corpus_dir, exist_ok=True)
        
        queue_dir = f"{active_dir}/out/default/queue"
        
        copied_count = 0
        if os.path.exists(queue_dir):
            for file_name in os.listdir(queue_dir):
                source_file = os.path.join(queue_dir, file_name)
                if os.path.isfile(source_file):
                    dest_file = os.path.join(corpus_dir, file_name)
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
        
        logger.info(f"Copied {copied_count} files to corpus directory.")
        
        if os.path.exists(corpus_dir):
            files = os.listdir(corpus_dir)
            logger.info(f"corpus directory contents: {len(files)} files")
            for file_name in files[:5]:
                logger.info(f"  - {file_name}")
            if len(files) > 5:
                logger.info(f"  ... and {len(files) - 5} more files")
        else:
            logger.warning("Failed to verify corpus directory")
        
        logger.info("Step 2: Running coverage measurement...")
        
        if not os.path.exists(corpus_dir):
            logger.error(f"corpus directory does not exist.")
            return False
        
        corpus_files = [f for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f))]
        if not corpus_files:
            logger.warning(f"No files in corpus directory.")
            return 0.0
        
        logger.info(f"Processing {len(corpus_files)} corpus files.")
        
        # Execute all files at once using corpus/* pattern (removed loop)
        corpus_glob_path = (
            f"/afk/library/{library_name}/harness_{run_index:03d}/"
            f"{category_name}/{gen_dir}/corpus/*"
        )
        
        # Generate coverage binary name (replace _fuzzer with _cov in fuzzer_name)
        coverage_bin_name = fuzzer_name.replace('_fuzzer', '_cov')
        fuzz_bin_path = (
            f"/afk/library/{library_name}/harness_{run_index:03d}/"
            f"{category_name}/{gen_dir}/{coverage_bin_name}"
        )
        profraw_path = (
            f"/afk/library/{library_name}/harness_{run_index:03d}/"
            f"{category_name}/{gen_dir}/coverage.profraw"
        )

        exec_cmd = (
            f"/bin/sh -lc '"
            f"cd /afk/library/{library_name}/harness_{run_index:03d}/{category_name}/{gen_dir} && "
            f"export LLVM_PROFILE_FILE=coverage.profraw && "
            f"./{coverage_bin_name} corpus/*"
            f"'"
        )
        try:
            exit_code, output = await asyncio.wait_for(
                asyncio.to_thread(
                    container.exec_run,
                    exec_cmd
                ),
                timeout=600
            )
        except asyncio.TimeoutError:
            logger.error(f"Coverage measurement timed out after 30 minutes.")
            try:
                container.exec_run(f"pkill -f {coverage_bin_name}", detach=True)
            except:
                pass
            return None
        logger.info(f"Coverage measurement result - exit_code: {exit_code}")
        if exit_code != 0:
            logger.error(f"Coverage measurement failed output: {output.decode() if output else ''}")
        exit_code_check, output_check = container.exec_run(f"ls -la {profraw_path}")
        logger.info(f"profraw file check: {output_check.decode() if output_check else 'No profraw file found'}")
        
        if exit_code_check != 0:
            logger.error(f"profraw file was not created.")
            return None
        
        profdata_path = (
            f"/afk/library/{library_name}/harness_{run_index:03d}/"
            f"{category_name}/{gen_dir}/coverage.profdata"
        )
        
        profdata_cmd = (
            f"llvm-profdata merge {profraw_path} "
            f"-o {profdata_path}"
        )
        exit_code, output = await asyncio.to_thread(container.exec_run, profdata_cmd)
        if exit_code != 0:
            logger.error(f"Profile data merge failed: {output.decode() if output else ''}")
            return None
        report_path = (
            f"/afk/library/{library_name}/harness_{run_index:03d}/"
            f"{category_name}/{gen_dir}/coverage_report.txt"
        )
        ignore_flag = build_coverage_ignore_regex(library_name)
        cov_report_cmd = (
            f"bash -c 'llvm-cov report {fuzz_bin_path} "
            f"--instr-profile={profdata_path} {ignore_flag} > {report_path}'"
        )
        exit_code, output = await asyncio.to_thread(container.exec_run, cov_report_cmd)
        if exit_code != 0:
            logger.error(f"Coverage report generation failed: {output.decode() if output else ''}")
            return None
        host_report_path = os.path.join(active_dir, "coverage_report.txt")
        metrics = parse_totals_from_text_report(host_report_path)
        if not metrics:
            logger.warning("[Coverage Summary] Failed to extract metrics from text report.")
            return None

        cov_percent = calculate_coverage_percentage(metrics)
        logger.info(f"=== Coverage measurement completed for {library_name}:{fuzzer_name} -> {cov_percent:.2f}% ===")
        return cov_percent
    except Exception as e:
        logger.error(f"Exception during coverage measurement: {e}")
        return None

async def check_fuzz_done(fuzzer_name: str, duration_seconds: int = 10) -> bool:
    """
    Check if afl-fuzz process is still running inside docker container.
    Wait for duration_seconds first, then check every 10 seconds with max wait time to prevent infinite loop.
    """
    container_name = f"{fuzzer_name}_container"
    max_wait_time = duration_seconds + 3600
    start_time = time.time()
    
    try:
        client = docker.from_env()
        try:
            container = client.containers.get(container_name)
        except docker.errors.NotFound:
            logger.warning(f"Container {container_name} does not exist.")
            return True
        
        logger.info(f"Fuzzing wait: waiting {duration_seconds} seconds...")
        await asyncio.sleep(duration_seconds)
        
        check_count = 0
        while time.time() - start_time < max_wait_time:
            check_count += 1
            
            try:
                exit_code, output = container.exec_run("pgrep afl-fuzz")
                if exit_code == 0:
                    elapsed = int(time.time() - start_time)
                    await asyncio.sleep(10)
                else:
                    elapsed = int(time.time() - start_time)
                    if elapsed < 180: # Termination under 180s is considered a harness defect
                        logger.warning(f"afl-fuzz process terminated prematurely in {elapsed}s. Harness regeneration needed.")
                        return False
                    else:
                        logger.info(f"afl-fuzz process has terminated. (elapsed: {elapsed}s)")
                    return True
            except Exception as check_error:
                logger.warning(f"Process check failed: {check_error}")
                try:
                    container.reload()
                    if container.status != 'running':
                        logger.info(f"Container has terminated: {container.status}")
                        return True
                except:
                    logger.info("Failed to check container status, assuming terminated")
                    return True
                await asyncio.sleep(10)
        
        logger.warning(f"Max wait time ({max_wait_time} seconds) exceeded, assuming force terminated")
        return True
        
    except Exception as e:
        logger.error(f"check_fuzz_done exception: {e}")
        return True

def clean_llm_response(response: str) -> str:
    """Remove unnecessary parts from LLM response and clean up."""
    response_clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response_clean = response_clean.replace('<think>', '').replace('</think>', '')
    return response_clean

def extract_json_from_response(response: str) -> dict:
    """Extract and clean JSON from LLM response."""
    logger.info(f"Starting JSON extraction - response length: {len(response)}")
    logger.info(f"Response start: {response[:200]}...")
    
    # First, try to find JSON inside code block
    code_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    code_block_match = re.search(code_block_pattern, response, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
        logger.info(f"JSON found in code block: {json_str}")
        try:
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Code block JSON parse failed: {e}")
            logger.error(f"Failed JSON string: {json_str}")
    
    start_idx = response.find('{')
    end_idx = response.rfind('}') + 1
    if start_idx != -1 and end_idx != 0:
        json_str = response[start_idx:end_idx]
        logger.info(f"JSON found in plain text: {json_str}")
        try:
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Plain JSON parse failed: {e}")
            logger.error(f"Failed JSON string: {json_str}")
    
    logger.error("Cannot find JSON in LLM response")
    logger.error(f"Full response: {response[:500]}...")
    return {}

def extract_flow_from_response(response: str) -> list:
    """Extract and clean JSON from LLM response."""
    logger.info(f"JSON extraction start - response length: {len(response)}")
    logger.info(f"Response start: {response[:200]}...")
    api_flow_list=[]
    try:
    # 1) JSON parsing
        flow_json = json.loads(response)

        # 2) Check for 'flow' key
        if "flow" not in flow_json:
            raise ValueError("Missing 'flow' key in LLM response")

        # 3) Check 'flow' type
        if not isinstance(flow_json["flow"], list):
            raise ValueError("'flow' must be a list")

        raw_list = flow_json["flow"]
        normalized_list = []
        for item in raw_list:
            if isinstance(item, str):
                try:
                    normalized_list.append(json.loads(item))
                except Exception as e:
                    logger.warning(f"Failed to parse stringified flow item: {item}. Error: {e}")
                    normalized_list.append({"api": item, "step": len(normalized_list)+1, "note": "Unparsed flow item"})
            elif isinstance(item, dict):
                normalized_list.append(item)
            else:
                normalized_list.append(item)
        api_flow_list = normalized_list

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON from LLM response: {e}")
        api_flow_list = []

    except ValueError as e:
        print(f"[ERROR] Invalid flow structure: {e}")
        api_flow_list = []

    except Exception as e:
        print(f"[ERROR] Unexpected error while processing api_flow: {e}")
        api_flow_list = []
    
    return api_flow_list

def extract_code_from_response(response: str) -> str:
    """Extract C/CPP code from LLM response.
    - Prioritize code blocks with language tags like ```c, ```cpp, ```c++, ```cxx, ```cc
    - Also extract code blocks without language tags if they have C/C++ characteristics (#include, extern "C", LLVMFuzzerTestOneInput, etc.)
    - Extract unfenced plain text code using heuristics
    """
    try:
        # 1) Search for code blocks with language tags (use the first valid one among multiple blocks)
        fenced_pattern = r"```[ \t]*([^\n]*)\n(.*?)\n```"
        matches = list(re.finditer(fenced_pattern, response, re.DOTALL))
        if matches:
            preferred_langs = {"c", "cpp", "c++", "cxx", "cc"}
            for m in matches:
                lang = (m.group(1) or "").strip().lower()
                code = (m.group(2) or "").strip()
                # If language tag is in preferred set, use it immediately
                if any(pl in lang for pl in preferred_langs):
                    logger.info("Code found in C/CPP language-tagged code block")
                    return code
            # 2) If no language tag or a different tag, but C/C++ characteristics are clear, adopt it
            c_like_markers = ("#include", "extern \"c\"", "LLVMFuzzerTestOneInput", "#define", ";\n")
            for m in matches:
                lang = (m.group(1) or "").strip().lower()
                code = (m.group(2) or "").strip()
                if any(marker.lower() in code.lower() for marker in c_like_markers):
                    logger.info("Code found in code block without language tag but with C/CPP characteristics")
                    return code

        # 3) If no fences at all: find start position using C/C++ characteristics and extract to the end
        inline_markers = [
            r"#include\s+[<\"]",
            r"extern\s+\"C\"",
            r"LLVMFuzzerTestOneInput\s*\(",
            r"int\s+main\s*\(",
        ]
        for pat in inline_markers:
            m = re.search(pat, response, re.IGNORECASE)
            if m:
                start_idx = m.start()
                code = response[start_idx:].strip()
                logger.info("Detected unfenced C/CPP code - extracted using heuristics")
                return code

        # 4) Final failure
        logger.error("Cannot find C/CPP code block in LLM response")
        logger.error(f"Full response: {response[:500]}...")
        return ""
    except Exception as e:
        logger.error(f"Exception during code extraction: {e}")
        logger.error(f"Full response: {response[:500]}...")
        return ""


def parse_fuzzer_stats(stats_file_path: str) -> dict:
    """
    Parse fuzzer_stats file and return as dictionary.
    """
    stats = {}
    try:
        if not os.path.exists(stats_file_path):
            logger.warning(f"fuzzer_stats file does not exist: {stats_file_path}")
            return stats
                    
        with open(stats_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        if '.' in value:
                            stats[key] = float(value.replace('%', ''))
                        else:
                            stats[key] = int(value)
                    except ValueError:
                        stats[key] = value
        
        return stats
        
    except Exception as e:
        logger.error(f"fuzzer_stats file path: {stats_file_path}")
        logger.error(f"fuzzer_stats parse failed: {e}")
        return {}

def should_regenerate_harness(fuzzer_output_dir: str, runtime_minutes: int = 10, category: str = "Unknown") -> tuple:
    """
    Analyze AFL++ output results to determine if harness regeneration is needed.
    
    Args:
        fuzzer_output_dir: AFL++ output directory path
        runtime_minutes: Runtime in minutes
        category: Category name (for logging)
    
    Returns:
        (regeneration_needed, reason)
    """
    logger.info(f"[Harness Evaluation][{category}] Starting evaluation - output directory: {fuzzer_output_dir}")
    
    stats_file = os.path.join(fuzzer_output_dir, "out", "default", "fuzzer_stats")
    stats = parse_fuzzer_stats(stats_file)
    
    if not stats:
        logger.error(f"[Harness Evaluation][{category}] Cannot read fuzzer_stats file: {stats_file}")
        return True, "Cannot read fuzzer_stats file"
    
    execs_done = stats.get("execs_done", 0)
    stability = stats.get("stability", 0.0)
    bitmap_cvg = stats.get("bitmap_cvg", 0.0)
    peak_rss_mb = stats.get("peak_rss_mb", 0)
    total_crashes = stats.get("total_crashes", 0)
    saved_crashes = stats.get("saved_crashes", 0)
    corpus_count = stats.get("corpus_count", 0)
    corpus_favored = stats.get("corpus_favored", 0)
    time_wo_finds = stats.get("time_wo_finds", 0)
    
    logger.info(f"[Harness Evaluation][{category}] Basic metrics - execs_done: {execs_done}, stability: {stability}%, "
                f"bitmap_cvg: {bitmap_cvg}%, peak_rss: {peak_rss_mb}MB")
    
    logger.info(f"[Harness Evaluation][{category}] 1. Stability check")
    if stability < 90.0:
        logger.warning(f"[Harness Evaluation][{category}] X Stability: {stability}% < 90%")
        return True, f"Stability too low: {stability}% < 90%"
    else:
        logger.info(f"[Harness Evaluation][{category}] OK Stability: {stability}% >= 90%")
    
    logger.info(f"[Harness Evaluation][{category}] 2. Coverage check")
    
    if bitmap_cvg <= 0.0:
        logger.warning(f"[Harness Evaluation][{category}] X Coverage: {bitmap_cvg}% <= 0.0%")
        return True, f"Coverage too low: {bitmap_cvg}% <= 0.0%"
    else:
        logger.info(f"[Harness Evaluation][{category}] OK Coverage: {bitmap_cvg}% > 0.0%")
    
    logger.info(f"[Harness Evaluation][{category}] 3. Crash rate check")
    logger.info(f"[Harness Evaluation][{category}] Crash info - total: {total_crashes}, unique: {saved_crashes}")
    logger.info(f"[Harness Evaluation][{category}] Corpus - count: {corpus_count}, favored: {corpus_favored}")
    
    crash_rate = total_crashes / execs_done if execs_done > 0 else 0
    logger.info(f"  - Total crashes: {total_crashes}, executions: {execs_done}")
    logger.info(f"  - Crash rate > 70%: {crash_rate > 0.7} (current: {crash_rate:.1%})")
    if total_crashes > execs_done * 0.7:
        logger.warning(f"[Harness Evaluation][{category}] X Immediate crash pattern detected (crash rate: {crash_rate:.1%})")
        return True, f"Immediate crash pattern: crashes={total_crashes}, rate={crash_rate:.1%}"
    logger.info(f"[Harness Evaluation][{category}] OK Crash check passed")


    logger.info(f"[Harness Evaluation][{category}] 4. Unique queue check")
    if corpus_count > 0:
        favored_ratio = (corpus_favored / corpus_count) * 100.0
        logger.info(f"[Harness Evaluation][{category}] unique queue metric - favored_ratio: {favored_ratio:.1f}%")
        if favored_ratio < 1.0:
            logger.warning(f"[Harness Evaluation][{category}] X Insufficient unique queue: favored_ratio {favored_ratio:.1f}% < 1.0%")
            return True, f"Insufficient unique queue (favored_ratio {favored_ratio:.1f}% < 1%)"
    
    logger.info(f"[Harness Evaluation][{category}] All checks passed")
    logger.info(f"[Harness Evaluation][{category}] Conclusion: Recommended for long fuzzing")
    return False, "Basic operation possible"

def remove_docker_container(fuzzer_name: str) -> bool:
    """
    Clean up Docker container (stop & remove) - including timeout and force cleanup.
    """
    container_name = f"{fuzzer_name}_container"
    logger.info(f"=== Docker container cleanup start for {container_name} ===")
    
    try:
        client = docker.from_env()
        
        try:
            container = client.containers.get(container_name)
            logger.info(f"Container {container_name} found")
            
            if container.status == 'running':
                logger.info(f"Stopping container {container_name}... (timeout: 60s)")
                try:
                    container.stop(timeout=60)
                    logger.info(f"Container {container_name} stopped")
                except Exception as stop_error:
                    logger.warning(f"Normal stop failed, attempting force stop: {stop_error}")
                    try:
                        result = subprocess.run(["docker", "kill", container_name], 
                                             capture_output=True, timeout=60, text=True)
                        if result.returncode == 0:
                            logger.info(f"Container {container_name} force stopped")
                        else:
                            logger.warning(f"Force stop failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        logger.error(f"Container force stop timeout: {container_name}")
                    except Exception as kill_error:
                        logger.error(f"Container force stop failed: {kill_error}")
            else:
                logger.info(f"Container {container_name} is already stopped")
            
            logger.info(f"Removing container {container_name}...")
            try:
                container.remove(force=True)
                logger.info(f"Container {container_name} removed")
            except Exception as remove_error:
                logger.warning(f"API remove failed, attempting subprocess force remove: {remove_error}")
                try:
                    result = subprocess.run(["docker", "rm", "-f", container_name], 
                                         capture_output=True, timeout=60, text=True)
                    if result.returncode == 0:
                        logger.info(f"Container {container_name} force removed")
                    else:
                        logger.warning(f"Force remove failed: {result.stderr}")
                except Exception as force_remove_error:
                    logger.error(f"Force remove also failed: {force_remove_error}")
                    return False
            
            return True
            
        except docker.errors.NotFound:
            logger.info(f"Container {container_name} does not exist (already removed)")
            return True
            
    except Exception as e:
        logger.error(f"Exception during container cleanup: {e}")
        try:
            logger.info(f"Final force cleanup attempt via subprocess: {container_name}")
            subprocess.run(["docker", "rm", "-f", container_name], 
                         capture_output=True, timeout=60, text=True)
            logger.info(f"Final force cleanup complete: {container_name}")
            return True
        except Exception as final_cleanup_error:
            logger.error(f"Final force cleanup also failed: {final_cleanup_error}")
            return False
    
    finally:
        logger.info(f"=== Docker container cleanup completed for {container_name} ===")

def move_successful_harness(library: str, category: str, run_index: int, gen_try: int, fix_try: int) -> bool:
    """
    Copy all files from successful gen{N}_fix{M} directory to parent directory.
    
    Args:
        library: Library name (e.g., libpng)
        category: Category name (e.g., Cleanup)
        run_index: Run index
        gen_try: Generation attempt number
        fix_try: Fix attempt number
    
    Returns:
        Success status
    """
    try:
        safe_category = category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        category_root = os.path.join("library", library, f"harness_{run_index:03d}", safe_category)
        gen_dir = f"gen{gen_try}_fix{fix_try}"
        source_dir = os.path.join(category_root, gen_dir)
        
        logger.info(f"=== Promoting successful attempt: {gen_dir} -> parent directory ===")
        
        if not os.path.exists(source_dir):
            logger.error(f"Source directory does not exist: {source_dir}")
            return False
        
        os.makedirs(category_root, exist_ok=True)
        
        promoted_count = 0
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            dest_path = os.path.join(category_root, item)
            
            try:
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"File promoted: {item}")
                    promoted_count += 1
                elif os.path.isdir(source_path):
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                    logger.info(f"Directory promoted: {item}")
                    promoted_count += 1
            except Exception as copy_error:
                logger.warning(f"File promotion failed: {item} - {copy_error}")
        
        logger.info(f"Promotion complete: {promoted_count} items copied to parent directory")
        return True
        
    except Exception as e:
        logger.error(f"Exception during promotion: {e}")
        logger.exception("Detailed exception info:")
        return False

def save_attempt_log(
    library: str,
    category: str, 
    gen_try: int,
    fix_try: int,
    result: str,
    error: str = "",
    coverage_score: float = 0.0,
    run_index: int = 0
) -> bool:
    """
    Save log for each gen_fix attempt.
    
    Args:
        library: Library name (e.g., libpng)
        category: Category name (e.g., Cleanup)
        gen_try: Generation attempt number
        fix_try: Fix attempt number
        result: Result ("build_failed", "evaluation_failed", "success")
        error: Build error or evaluation failure reason
        coverage_score: Coverage score
    
    Returns:
        Save success status
    """
    try:
        safe_category = category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        
        harness_root = os.path.join("library", library, f"harness_{run_index:03d}", safe_category)
        gen_dir = f"gen{gen_try}_fix{fix_try}"
        log_dir = os.path.join(harness_root, gen_dir)
        
        os.makedirs(log_dir, exist_ok=True)
        
        filename = "attempt.json"
        log_path = os.path.join(log_dir, filename)
        
        log_data = {
            "library": library,
            "category": category,
            "gen_try": gen_try,
            "fix_try": fix_try,
            "result": result,
            "coverage_score": coverage_score,
            "error": error
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Attempt log saved: {log_path}")
        return True
        
    except Exception as e:
        logger.error(f"Attempt log save failed: {e}")
        return False

def cleanup_all_fuzzer_containers(library_name: str = None) -> bool:
    """
    Clean up all fuzzer-related containers.
    Args:
        library_name: Only clean containers for specific library (None for all fuzzer containers)
    """
    logger.info("=== Global fuzzer container cleanup start ===")
    success_count = 0
    total_count = 0
    
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
        
        for container in containers:
            container_name = container.name
            
            if "fuzzer_container" in container_name:
                if library_name and library_name not in container_name:
                    continue
                    
                total_count += 1
                logger.info(f"Cleaning up: {container_name} (status: {container.status})")
                
                try:
                    if container.status == 'running':
                        try:
                            container.stop(timeout=60)
                            logger.info(f"Container stopped: {container_name}")
                        except Exception as stop_error:
                            logger.warning(f"Normal stop failed, force stopping: {container_name} - {stop_error}")
                            try:
                                subprocess.run(["docker", "kill", container_name], 
                                             capture_output=True, timeout=60, text=True)
                            except:
                                pass
                    
                    container.remove(force=True)
                    logger.info(f"Container removed: {container_name}")
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Container cleanup failed: {container_name} - {e}")
                    try:
                        subprocess.run(["docker", "rm", "-f", container_name], 
                                     capture_output=True, timeout=60, text=True)
                        logger.info(f"Force remove complete: {container_name}")
                        success_count += 1
                    except Exception as force_error:
                        logger.error(f"Force remove also failed: {container_name} - {force_error}")
        
        logger.info(f"Global container cleanup complete: {success_count}/{total_count} succeeded")
        return success_count == total_count
                    
    except Exception as e:
        logger.error(f"Exception during global container cleanup: {e}")
        return False
    
    finally:
        logger.info("=== Global fuzzer container cleanup completed ===")


# ============================================================================
# Coverage-based harness selection functions
# ============================================================================

def save_json(path: str, data: dict):
    """Save JSON file - compress unused_apis, selected_categories to single line"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    # Compress long list keys to single line
    for key in ['unused_apis', 'selected_categories']:
        pattern = rf'"{key}":\s*\[\s*(.*?)\s*\]'
        
        def compact_match(match):
            content = match.group(1)
            compacted = re.sub(r'\s*\n\s*', ' ', content)
            compacted = re.sub(r',\s*', ', ', compacted).strip()
            return f'"{key}": [{compacted}]'
        
        json_str = re.sub(pattern, compact_match, json_str, flags=re.DOTALL)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def load_json(path: str) -> dict:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_coverage_metrics(profdata_path: str, binary_path: str, library_name: str = "") -> dict:
    """
    Extract 4 coverage metrics using llvm-cov export
    
    Args:
        profdata_path: .profdata file path
        binary_path: Coverage measurement binary path
        library_name: Library name (for coverage exclusion patterns)
    
    Returns:
        {
            'regions': {'covered': int, 'total': int},
            'functions': {'covered': int, 'total': int},
            'lines': {'covered': int, 'total': int},
            'branches': {'covered': int, 'total': int}
        }
    """
    cmd = [
        "llvm-cov", "export", binary_path,
        f"--instr-profile={profdata_path}",
        "--skip-expansions",
        "--summary-only"
    ]
    
    # Add library-specific exclusion patterns (e.g., exclude external dependencies like zlib)
    if library_name:
        exclude_patterns = get_coverage_exclude_patterns(library_name)
        if exclude_patterns:
            regex_pattern = "|".join([f".*{p}.*" for p in exclude_patterns])
            cmd.append(f"--ignore-filename-regex=({regex_pattern})")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"llvm-cov export failed: {result.stderr}")
            return None
        
        coverage_data = json.loads(result.stdout)
        summary = coverage_data['data'][0]['totals']
        
        return {
            'regions': {
                'covered': summary['regions']['covered'],
                'total': summary['regions']['count']
            },
            'functions': {
                'covered': summary['functions']['covered'],
                'total': summary['functions']['count']
            },
            'lines': {
                'covered': summary['lines']['covered'],
                'total': summary['lines']['count']
            },
            'branches': {
                'covered': summary['branches']['covered'],
                'total': summary['branches']['count']
            }
        }
    except Exception as e:
        logger.error(f"Coverage metrics extraction failed: {e}")
        return None


def calculate_coverage_percentage(metrics: dict) -> float:
    """
    Calculate total coverage based on branches
    """
    branches = metrics['branches']
    return (branches['covered'] / branches['total']) * 100 if branches['total'] > 0 else 0.0


def parse_totals_from_text_report(report_path: str) -> Optional[dict]:
    """
    Parse the TOTAL row from llvm-cov report text report (coverage_report.txt)
    and return as a metrics dictionary in the same format as current_metrics.json.
    """
    if not os.path.exists(report_path):
        logger.warning(f"Text report file does not exist: {report_path}")
        return None

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_line = None
        for line in lines:
            if line.strip().startswith("TOTAL"):
                total_line = line
                break

        if not total_line:
            logger.warning(f"TOTAL row not found in text report: {report_path}")
            return None

        parts = total_line.split()
        # Format example:
        # TOTAL 89888 89688 0.22% 4250 4232 0.42% 115923 115710 0.18% 51216 51118 0.19%
        if len(parts) < 13:
            logger.warning(f"TOTAL row parsing failed (insufficient tokens): {total_line.strip()}")
            return None

        def _to_int(value: str) -> int:
            return int(value.replace(',', ''))

        regions_total = _to_int(parts[1])
        regions_missed = _to_int(parts[2])
        functions_total = _to_int(parts[4])
        functions_missed = _to_int(parts[5])
        lines_total = _to_int(parts[7])
        lines_missed = _to_int(parts[8])
        branches_total = _to_int(parts[10])
        branches_missed = _to_int(parts[11])

        metrics = {
            'regions': {
                'covered': regions_total - regions_missed,
                'total': regions_total,
            },
            'functions': {
                'covered': functions_total - functions_missed,
                'total': functions_total,
            },
            'lines': {
                'covered': lines_total - lines_missed,
                'total': lines_total,
            },
            'branches': {
                'covered': branches_total - branches_missed,
                'total': branches_total,
            },
        }

        logger.info(
            f"[Text Report TOTAL] regions: {metrics['regions']}, "
            f"functions: {metrics['functions']}, "
            f"lines: {metrics['lines']}, "
            f"branches: {metrics['branches']}"
        )
        return metrics
    except Exception as e:
        logger.error(f"Exception during text report TOTAL parsing: {e}")
        return None

def get_harness_profdata(harness_path: str) -> str:
    """
    Find profdata file in harness directory
    
    Args:
        harness_path: Harness directory path (e.g., library/sqlite3/harness_000/URI)
    
    Returns:
        profdata file path (None if not found)
    """
    profdata_path = os.path.join(harness_path, "coverage.profdata")
    if os.path.exists(profdata_path):
        logger.info(f"  profdata file found: {profdata_path}")
        return profdata_path
    else:
        logger.warning(f"  profdata file missing: {profdata_path}")
        return None


def get_profdata_missing_reason(category_path: str) -> str:
    """
    When profdata is missing from the category folder, check genX_fixY folders' attempt.json
    to find the actual failure reason.
    
    Args:
        category_path: Category directory path (e.g., library/libpng/harness_000/Decoding)
    
    Returns:
        Failure reason string
    """
    import glob
    
    # Find genX_fixY folders
    gen_dirs = sorted(glob.glob(os.path.join(category_path, "gen*_fix*")), reverse=True)
    
    if not gen_dirs:
        return "No harness generation attempt"
    
    # Check from most recent attempt
    for gen_dir in gen_dirs:
        attempt_path = os.path.join(gen_dir, "attempt.json")
        
        if not os.path.exists(attempt_path):
            continue
        
        try:
            with open(attempt_path, 'r', encoding='utf-8') as f:
                attempt = json.load(f)
            
            result = attempt.get("result", "")
            error = attempt.get("error", "")
            llm_judgement = attempt.get("llm_judgement", {})
            
            # Build failure
            if "build_failed" in result or error:
                return f"Build failed: {error[:100] if error else result}"
            
            # LLM judgment rejected
            if llm_judgement:
                should_continue = llm_judgement.get("should_continue", True)
                if not should_continue:
                    reason = llm_judgement.get("reason", "")
                    crash_count = llm_judgement.get("crash_count", 0)
                    if isinstance(reason, dict):
                        reason = json.dumps(reason, ensure_ascii=False)[:100]
                    elif isinstance(reason, str):
                        reason = reason[:100]
                    return f"LLM rejected ({crash_count} crashes): {reason}"
            
            # Quality evaluation failed
            if "quality_failed" in result:
                return f"Quality check failed: {result}"
            
            # profdata exists in genX_fixY but not promoted
            gen_profdata = os.path.join(gen_dir, "coverage.profdata")
            if os.path.exists(gen_profdata):
                return f"Not promoted ({os.path.basename(gen_dir)} has profdata)"
            
        except Exception as e:
            continue
    
    return "Profdata generation failed (unknown reason)"


def find_harness_source_file(category_path: str, safe_category: str) -> Optional[str]:
    """
    Find harness source code. Use promoted files first, otherwise find the latest from gen directory.
    """
    candidate_names = [
        f"{safe_category}_harness.c",
        f"{safe_category}_harness.cc",
        f"{safe_category}_harness.cpp"
    ]
    
    for name in candidate_names:
        candidate = os.path.join(category_path, name)
        if os.path.exists(candidate):
            return candidate
    
    try:
        gen_dirs = sorted(
            [
                d for d in os.listdir(category_path)
                if d.startswith("gen") and os.path.isdir(os.path.join(category_path, d))
            ],
            reverse=True
        )
    except FileNotFoundError:
        return None
    
    for gen_dir in gen_dirs:
        for name in candidate_names:
            candidate = os.path.join(category_path, gen_dir, name)
            if os.path.exists(candidate):
                return candidate
    
    return None
    
    for name in candidate_names:
        candidate = os.path.join(category_path, name)
        if os.path.exists(candidate):
            return candidate
    
    try:
        gen_dirs = sorted(
            [
                d for d in os.listdir(category_path)
                if d.startswith("gen") and os.path.isdir(os.path.join(category_path, d))
            ],
            reverse=True
        )
    except FileNotFoundError:
        return None
    
    for gen_dir in gen_dirs:
        for name in candidate_names:
            candidate = os.path.join(category_path, gen_dir, name)
            if os.path.exists(candidate):
                return candidate
    
    return None


def extract_coverage_metrics(container, profdata_path: str, binary_path: str, library_name: str = "") -> dict:
    """
    Extract 4 coverage metrics using llvm-cov export inside Docker container
    
    Args:
        container: Docker container object
        profdata_path: .profdata file path (host-based)
        binary_path: Coverage measurement binary path (host-based)
        library_name: Library name (for coverage exclusion patterns)
    
    Returns:
        Coverage metrics dictionary
    """
    try:
        # Convert to container internal paths
        profdata_container = f"/afk/{profdata_path}"
        binary_container = f"/afk/{binary_path}"
        
        # Use library-specific exclusion patterns (e.g., exclude external dependencies like zlib)
        ignore_flag = build_coverage_ignore_regex(library_name) if library_name else "--ignore-filename-regex=.*_harness\\.c"
        cmd = (
            f"llvm-cov export {binary_container} "
            f"--instr-profile={profdata_container} "
            f"--skip-expansions --summary-only "
            f"{ignore_flag}"
        ).strip()
        
        exit_code, output = container.exec_run(cmd)
        if exit_code != 0:
            logger.error(f"llvm-cov export failed: {output.decode() if output else ''}")
            return None
        
        # JSON parsing
        coverage_data = json.loads(output.decode())
        summary = coverage_data['data'][0]['totals']
        
        return {
            'regions': {
                'covered': summary['regions']['covered'],
                'total': summary['regions']['count']
            },
            'functions': {
                'covered': summary['functions']['covered'],
                'total': summary['functions']['count']
            },
            'lines': {
                'covered': summary['lines']['covered'],
                'total': summary['lines']['count']
            },
            'branches': {
                'covered': summary['branches']['covered'],
                'total': summary['branches']['count']
            }
        }
    except Exception as e:
        logger.error(f"Coverage metrics extraction failed: {e}")
        return None


def generate_final_report(container, profdata_path: str, library_binary_path: str, coverage_dir: str, library_name: str = ""):
    """
    Generate final coverage report
    
    Args:
        container: Docker container object
        profdata_path: merged.profdata path (host-based)
        library_binary_path: Library binary path (host-based)
        coverage_dir: final_harnesses directory path (host-based)
        library_name: Library name (for coverage exclusion patterns)
    """
    try:
        if not library_binary_path or not os.path.exists(library_binary_path):
            logger.warning("Cannot find library binary, skipping report generation.")
            return
        
        profdata_container = f"/afk/{profdata_path}"
        binary_container = f"/afk/{library_binary_path}"
        
        # Apply library-specific coverage exclusion patterns (e.g., exclude external dependencies like zlib)
        ignore_flag = build_coverage_ignore_regex(library_name) if library_name else "--ignore-filename-regex='.*_harness\\.c'"
        
        # llvm-cov report (text)
        report_path = f"{coverage_dir}/coverage_report.txt"
        report_cmd = (
            f"llvm-cov report {binary_container} "
            f"--instr-profile={profdata_container} "
            f"{ignore_flag}"
        )
        
        exit_code, output = container.exec_run(report_cmd)
        if exit_code == 0:
            with open(report_path, 'w') as f:
                f.write(output.decode())
            logger.info(f"Report generated: {report_path}")
        else:
            logger.warning(f"llvm-cov report execution failed: {output.decode() if output else ''}")

    except Exception as e:
        logger.error(f"Error during report generation: {e}")


def filter_and_combine_coverage(lib: str, categorized_apis: dict, current_run_index: int, unused_apis: list = None) -> float:
    """
    Filter harnesses from current run based on coverage and calculate final coverage
    Replaces existing combine_all_coverage()
    
    Args:
        lib: Library name
        categorized_apis: Dictionary of APIs per category
        current_run_index: Current run index
    
    Returns:
        Final coverage percentage
    """
    coverage_dir = f"library/{lib}/final_harnesses"
    merged_profdata = f"{coverage_dir}/merged.profdata"
    current_metrics_file = f"{coverage_dir}/current_metrics.json"
    
    # Create directory
    os.makedirs(coverage_dir, exist_ok=True)
    
    # Create Docker container
    container_name = f"{lib}_coverage_merge"
    client = docker.from_env()
    container = None
    
    try:
        # Remove existing container if present
        try:
            old_container = client.containers.get(container_name)
            old_container.remove(force=True)
            logger.info(f"Removed existing container: {container_name}")
        except docker.errors.NotFound:
            pass
        
        # Create new container
        logger.info(f"Creating coverage measurement container: {container_name}")
        container = client.containers.run(
            image="afk",
            name=container_name,
            detach=True,
            tty=True,
            network_mode="none",
            volumes=[
                f"{HOST_BASE_PATH}/library:/afk/library:rw"
            ],
            working_dir="/afk"
        )
        logger.info(f"Container created: {container_name}")
        
        # No separate baseline generation needed, first harness's profdata is used as-is
        if not os.path.exists(merged_profdata) or not os.path.exists(current_metrics_file):
            # Initialize empty metrics (updated when first harness is selected)
            baseline_metrics = {
                'regions': {'covered': 0, 'total': 0},
                'functions': {'covered': 0, 'total': 0},
                'lines': {'covered': 0, 'total': 0},
                'branches': {'covered': 0, 'total': 0}
            }
            save_json(current_metrics_file, baseline_metrics)
            logger.info("Baseline metrics initialized (profdata created when first harness is selected)")
        
        # Load current metrics
        current_metrics = load_json(current_metrics_file)
        
        # Load selection_log (existing selection records)
        selection_log_path = f"{coverage_dir}/selection_log.json"
        if os.path.exists(selection_log_path):
            selection_log = load_json(selection_log_path)
        else:
            selection_log = {}
        
        selected_count = 0
        rejected_count = 0
        first_selected_cov_binary = None  # For final report generation
        
        logger.info("=" * 60)
        logger.info(f"Starting harness selection (Run {current_run_index})")
        logger.info("=" * 60)
        
        library_base = f"library/{lib}"
        run_dir = f"harness_{current_run_index:03d}"
        run_path = os.path.join(library_base, run_dir)
        run_output_dir = os.path.join(coverage_dir, run_dir)
        os.makedirs(run_output_dir, exist_ok=True)
        
        if not os.path.exists(run_path):
            logger.warning(f"Run directory not found: {run_path}")
            return calculate_coverage_percentage(current_metrics) if current_metrics['regions']['total'] > 0 else 0.0
        
        # Check if merged.profdata exists (for determining first harness)
        is_first_harness = not os.path.exists(merged_profdata)
        
        # Iterate through categories in current run
        for category in categorized_apis.keys():
            safe_category = category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            category_path = os.path.join(run_path, safe_category)
            
            if not os.path.isdir(category_path):
                logger.warning(f"Category directory not found: {category_path}")
                continue
            
            logger.info(f"\nEvaluating: {run_dir}/{category}")
            
            # Find profdata file
            harness_profdata = get_harness_profdata(category_path)
            if not harness_profdata:
                # Determine actual failure reason
                missing_reason = get_profdata_missing_reason(category_path)
                logger.warning(f"  {missing_reason}")
                rejected_count += 1
                selection_log[f"{run_dir}/{category}"] = {
                    "selected": False,
                    "reason": missing_reason
                }
                continue
            
            # Coverage binary path
            cov_binary = os.path.join(category_path, f"{lib}_{safe_category}_cov")
            
            if not os.path.exists(cov_binary):
                logger.warning(f"  Coverage binary not found: {cov_binary}")
                rejected_count += 1
                continue
            
            harness_source_path = find_harness_source_file(category_path, safe_category)
            if not harness_source_path:
                logger.warning(f"  Cannot find harness source code, skipping evaluation.")
                rejected_count += 1
                selection_log[f"{run_dir}/{safe_category}"] = {
                    "selected": False,
                    "reason": "harness source missing"
                }
                continue
            
            temp_profdata = os.path.join(
                coverage_dir,
                f"temp_{run_dir}_{safe_category}_{int(time.time()*1000)}.profdata"
            )
            
            # First harness only: copy to temp_profdata when merged.profdata doesn't exist, otherwise always merge
            if is_first_harness:
                # First harness: copy harness.profdata to temp_profdata (no merge needed)
                logger.info(f"  First harness - copying harness.profdata to temp.profdata (no merge needed)")
                shutil.copy(harness_profdata, temp_profdata)
                # Use merge from next iteration
                is_first_harness = False
            else:
                # Merge merged.profdata + harness.profdata -> temp (in container)
                merge_cmd = (
                    f"llvm-profdata merge "
                    f"/afk/{merged_profdata} /afk/{harness_profdata} "
                    f"-o /afk/{temp_profdata}"
                )
                
                logger.info(f"  profdata merge command: {merge_cmd}")
                logger.info(f"  Existing merged.profdata: {merged_profdata} (exists: {os.path.exists(merged_profdata)}, size: {os.path.getsize(merged_profdata) if os.path.exists(merged_profdata) else 0} bytes)")
                logger.info(f"  New harness.profdata: {harness_profdata} (exists: {os.path.exists(harness_profdata)}, size: {os.path.getsize(harness_profdata) if os.path.exists(harness_profdata) else 0} bytes)")
                
                exit_code, output = container.exec_run(merge_cmd)
                if exit_code != 0:
                    error_msg = output.decode() if output else 'No error message'
                    logger.error(f"  merged.profdata merge failed!")
                    logger.error(f"     Exit code: {exit_code}")
                    logger.error(f"     Error: {error_msg}")
                    logger.error(f"     Command: {merge_cmd}")
                    logger.error(f"     Merged profdata exists: {os.path.exists(merged_profdata)}")
                    logger.error(f"     Harness profdata exists: {os.path.exists(harness_profdata)}")
                    rejected_count += 1
                    continue
            
            # Verify temp_profdata creation (for both first and subsequent harnesses)
            if os.path.exists(temp_profdata):
                file_size = os.path.getsize(temp_profdata)
                logger.info(f"  temp.profdata created (size: {file_size} bytes)")
            else:
                logger.error(f"  temp.profdata file not created: {temp_profdata}")
                rejected_count += 1
                continue
            
            # Extract new metrics (in container, with library-specific exclusion patterns)
            new_metrics = extract_coverage_metrics(container, temp_profdata, cov_binary, library_name=lib)
            if not new_metrics:
                rejected_count += 1
                continue
            
            # Compare 4 metrics (select if any metric improves, same as promptfuzz)
            improvement = {
                'regions': new_metrics['regions']['covered'] - current_metrics['regions']['covered'],
                'functions': new_metrics['functions']['covered'] - current_metrics['functions']['covered'],
                'lines': new_metrics['lines']['covered'] - current_metrics['lines']['covered'],
                'branches': new_metrics['branches']['covered'] - current_metrics['branches']['covered']
            }
            
            any_improved = any(v > 0 for v in improvement.values())
            
            if any_improved:
                # Selected!
                logger.info(f"  Selected! (+regions:{improvement['regions']}, "
                           f"+functions:{improvement['functions']}, "
                           f"+lines:{improvement['lines']}, "
                           f"+branches:{improvement['branches']})")
                
                # Update merged.profdata
                shutil.copy(temp_profdata, merged_profdata)
                # Save first selected cov_binary (for final report generation)
                if first_selected_cov_binary is None:
                    first_selected_cov_binary = cov_binary
                old_metrics = {k: v.copy() if isinstance(v, dict) else v for k, v in current_metrics.items()}
                current_metrics = new_metrics
                save_json(current_metrics_file, current_metrics)
                
                # Save harness (keep per category within run directory)
                category_output_dir = os.path.join(run_output_dir, safe_category)
                os.makedirs(category_output_dir, exist_ok=True)
                
                shutil.copy(harness_source_path, os.path.join(category_output_dir, os.path.basename(harness_source_path)))
                shutil.copy(harness_profdata, os.path.join(category_output_dir, "coverage.profdata"))
                if os.path.exists(cov_binary):
                    shutil.copy(cov_binary, os.path.join(category_output_dir, os.path.basename(cov_binary)))
                
                # Copy reproduce binary, crash_reproduce, and crashes if they exist
                reproduce_bin_src = os.path.join(category_path, f"{lib}_{safe_category}_reproduce")
                if os.path.exists(reproduce_bin_src):
                    shutil.copy2(reproduce_bin_src, os.path.join(category_output_dir, os.path.basename(reproduce_bin_src)))
                    
                crash_reproduce_src = os.path.join(category_path, "crash_reproduce")
                if os.path.exists(crash_reproduce_src) and os.path.isdir(crash_reproduce_src):
                    dest_crash_reproduce = os.path.join(category_output_dir, "crash_reproduce")
                    if os.path.exists(dest_crash_reproduce):
                        shutil.rmtree(dest_crash_reproduce)
                    shutil.copytree(crash_reproduce_src, dest_crash_reproduce)
                    
                crashes_src = os.path.join(category_path, "out", "default", "crashes")
                if os.path.exists(crashes_src) and os.path.isdir(crashes_src):
                    dest_crashes = os.path.join(category_output_dir, "crashes")
                    if os.path.exists(dest_crashes):
                        shutil.rmtree(dest_crashes)
                    shutil.copytree(crashes_src, dest_crashes)
                
                log_key = f"{run_dir}/{safe_category}"
                
                # Log record (save only improvement - concise)
                selection_log[log_key] = {
                    "selected": True,
                    "improvement": improvement
                }
                
                selected_count += 1
            else:
                # Rejected
                reasons = []
                if improvement['regions'] <= 0:
                    reasons.append("regions")
                if improvement['functions'] <= 0:
                    reasons.append("functions")
                if improvement['lines'] <= 0:
                    reasons.append("lines")
                if improvement['branches'] <= 0:
                    reasons.append("branches")
                
                logger.info(f"  Rejected (no improvement: {', '.join(reasons)})")
                
                # Log record
                selection_log[f"{run_dir}/{safe_category}"] = {
                    "selected": False,
                    "reason": f"No improvement in: {', '.join(reasons)}",
                    "improvement": improvement
                }
                
                rejected_count += 1
            
            # Clean up temp_profdata (always delete regardless of selection)
            if os.path.exists(temp_profdata):
                try:
                    os.remove(temp_profdata)
                except Exception as e:
                    logger.warning(f"  temp.profdata deletion failed: {e}")
        
        # Save selection_log.json
        save_json(f"{coverage_dir}/selection_log.json", selection_log)
        
        # Generate final report (using selected _cov binary)
        if os.path.exists(merged_profdata) and first_selected_cov_binary:
            generate_final_report(container, merged_profdata, first_selected_cov_binary, coverage_dir, library_name=lib)

        # Synchronize current_metrics.json based on coverage_report.txt
        text_report_path = os.path.join(coverage_dir, "coverage_report.txt")
        text_metrics = parse_totals_from_text_report(text_report_path)
        if text_metrics:
            # Update current_metrics.json (current state of selection process)
            current_metrics = text_metrics
            save_json(current_metrics_file, current_metrics)
            
            # Save per-run history to coverage_history.json
            history_file = f"{coverage_dir}/coverage_history.json"
            if os.path.exists(history_file):
                coverage_history = load_json(history_file)
            else:
                coverage_history = {}
            
            # List of categories selected in this run
            selected_categories = [
                key.split('/')[-1] for key, val in selection_log.items()
                if key.startswith(run_dir) and val.get('selected')
            ]
            
            # Coverage percentage calculation helper
            def calc_pct(covered, total):
                return round(covered / total * 100, 2) if total > 0 else 0
            
            # Save cumulative coverage per run (harness_000, harness_001, ...)
            coverage_history[run_dir] = {
                'timestamp': datetime.datetime.now().isoformat(),
                'selected_categories': selected_categories,
                'unused_apis': unused_apis or [],
                'cumulative_coverage': {
                    'regions': {
                        'covered': text_metrics['regions']['covered'],
                        'total': text_metrics['regions']['total'],
                        'percent': calc_pct(text_metrics['regions']['covered'], text_metrics['regions']['total'])
                    },
                    'functions': {
                        'covered': text_metrics['functions']['covered'],
                        'total': text_metrics['functions']['total'],
                        'percent': calc_pct(text_metrics['functions']['covered'], text_metrics['functions']['total'])
                    },
                    'lines': {
                        'covered': text_metrics['lines']['covered'],
                        'total': text_metrics['lines']['total'],
                        'percent': calc_pct(text_metrics['lines']['covered'], text_metrics['lines']['total'])
                    },
                    'branches': {
                        'covered': text_metrics['branches']['covered'],
                        'total': text_metrics['branches']['total'],
                        'percent': calc_pct(text_metrics['branches']['covered'], text_metrics['branches']['total'])
                    }
                }
            }
            
            save_json(history_file, coverage_history)
        
        logger.info("\n" + "=" * 60)
        logger.info("Selection Results")
        logger.info("=" * 60)
        logger.info(f"Selected harnesses: {selected_count}")
        logger.info(f"Rejected harnesses: {rejected_count}")
        logger.info(f"Results saved: {coverage_dir}/")
        logger.info(f"Log file: {coverage_dir}/selection_log.json")
        logger.info("=" * 60)
        
        # Return final coverage %
        final_coverage = calculate_coverage_percentage(current_metrics)
        if selected_count == 0:
            logger.warning("No harnesses selected. Returning existing cumulative coverage.")
        logger.info(f"Final coverage: {final_coverage:.2f}%")
        return final_coverage
    
    finally:
        # Clean up container
        if container is not None:
            try:
                logger.info(f"Cleaning up container: {container_name}")
                container.stop()
                container.remove()
                logger.info(f"Container removed: {container_name}")
            except Exception as e:
                logger.warning(f"Error during container cleanup (ignored): {e}")

async def reproduce_crash(
    library: str,
    category: str, 
    run_index: int,
    gen_try: int,
    fix_try: int
) -> Tuple[list, list]:
    """
    Execute crash files directly on host using reproduce binary and collect output.
    Uses asyncio.gather for parallel processing to improve speed.
    
    Returns:
        (crash_files, crash_outputs): list of crash file paths, list of outputs for each crash
    """
    safe_category = re.sub(r'[^A-Za-z0-9_]', '_', category)
    gen_dir = f"gen{gen_try}_fix{fix_try}"
    base_path = os.path.join("library", library, f"harness_{run_index:03d}", safe_category, gen_dir)
    
    crash_dir = os.path.join(base_path, "out", "default", "crashes")
    reproduce_bin = os.path.join(base_path, f"{library}_{safe_category}_reproduce")
    
    if not os.path.exists(crash_dir) or not os.path.exists(reproduce_bin):
        if logger:
            logger.warning(f"Crash directory or reproduce binary not found: {crash_dir}, {reproduce_bin}")
        return [], []
    
    crash_file_list = [f for f in os.listdir(crash_dir) if f != "README.txt"]
    
    if not crash_file_list:
        if logger:
            logger.info(f"No crash files to reproduce.")
        return [], []
    
    async def reproduce_one(crash_file: str) -> Tuple[str, str]:
        """Reproduce single crash file directly on host"""
        crash_path = os.path.join(crash_dir, crash_file)
        
        # Execute directly via subprocess
        try:
            proc = await asyncio.create_subprocess_exec(
                reproduce_bin, crash_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            
            output = f"=== Crash: {crash_file} ===\n"
            output += f"STDOUT:\n{stdout.decode('utf-8', errors='replace')}\n"
            output += f"STDERR:\n{stderr.decode('utf-8', errors='replace')}\n"
            output += f"Return code: {proc.returncode}\n"
        except asyncio.TimeoutError:
            output = f"=== Crash: {crash_file} ===\nTIMEOUT: reproduce execution timed out (30 seconds)\n"
        except Exception as e:
            if logger:
                logger.error(f"Crash reproduce failed ({crash_file}): {e}")
            output = f"=== Crash: {crash_file} ===\nERROR: {e}\n"
        
        # Always save file (regardless of success/failure)
        # Save to crash_reproduce directory
        reproduce_dir = os.path.join(base_path, "crash_reproduce")
        os.makedirs(reproduce_dir, exist_ok=True)
        
        # id:000000,sig:06,... -> id_000000.txt
        id_part = crash_file.split(',')[0].replace(':', '_') if ',' in crash_file else crash_file.replace(':', '_')
        output_path = os.path.join(reproduce_dir, f"{id_part}.txt")
        try:
            with open(output_path, 'w') as f:
                f.write(output)
        except Exception as e:
            if logger:
                logger.warning(f"File save failed ({id_part}): {e}")
        
        return crash_path, output
    
    # Reproduce all crash files in parallel
    if logger:
        logger.info(f"Starting parallel reproduction of {len(crash_file_list)} crashes on host")
    
    results = await asyncio.gather(*[reproduce_one(f) for f in crash_file_list])
    
    crash_files = [r[0] for r in results]
    crash_outputs = [r[1] for r in results]
    
    if logger:
        logger.info(f"Crash reproduction complete: {len(crash_files)} files")
    
    return crash_files, crash_outputs

