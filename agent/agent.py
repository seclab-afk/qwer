import logging
import sys
import gc
import os
import json
import datetime
from typing import List, ClassVar, Optional, Union
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ConfigDict
from config import LLM_SERVER_URL, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME
from module import extract_lib_symbols, clone_and_build_library, load_build_config, generate_ast_files, clean_llm_response, extract_json_from_response, OllamaRAGManager, filter_and_combine_coverage, llm_semaphore, get_unused_apis, build_check, sync_image_dependencies
from harness_agent import run_parallel_harness_generation
import harness_agent
import module
import argparse
import asyncio
import braille
import glob
from zoneinfo import ZoneInfo
from logger_config import setup_unified_logger, get_logger, get_token_file
from dashboard_manager import dashboard_manager
import httpx
import re
import time




class LibraryAgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    current_step: str = "start"
    api_list: List[str] = Field(default_factory=list)
    build_log: str = ""
    build_success: bool = False
    api_extraction_success: bool = False
    categorized_apis: dict = Field(default_factory=dict)
    error: str = ""
    lib: str = ""
    lib_api_dir: str = ""
    lib_fuzz_dir: str = ""
    lib_cov_dir: str = ""
    ast_chunks_path: str =""
    response: str = ""
    total_coverage: float = 0.0
    coverage_threshold: float = 20.0
    current_run_index: int = 0  
    max_runs: int = 1000  
    rag_build_success: bool = False
    rag_query_result: dict = Field(default_factory=dict)
    rag_manager: Optional[OllamaRAGManager] = None
    retry_count: int = 0 
    max_retries: int = 100 
    unused_apis: List[str] = Field(default_factory=list) 
    max_time: Optional[int] = None
    stop_early: bool = False

def process_user_input(state: LibraryAgentState) -> LibraryAgentState:
    state.current_step = "build_library"
    return state

def build_library_node(state: LibraryAgentState) -> LibraryAgentState:
    try:
        logger.info(f"=== {state.lib_api_dir} build start ===")
        logger.info(f"curent working dir: {os.getcwd()}")
        print(state.lib_api_dir)
        if build_check(state.lib_api_dir):
            ok = clone_and_build_library(state.lib_api_dir)
            
        else:
            ok=True
        logger.info(f"build result: {'success' if ok else 'fail'}")
        if not ok:
            logger.error("=== build fail ===")
            state.build_success = False
            state.current_step = "respond"  # build fail -> respond
            return state
        state.build_success = True
        state.current_step = "extract_api"  # build success -> extract api
        return state
    except Exception as e:
        logger.error(f"build_library_node exception: {e}")
        state.error = f"[build_library_node] {e}"
        state.build_success = False
        state.current_step = "respond"
        return state

async def extract_api_node(state: LibraryAgentState) -> LibraryAgentState:
    try:
        if not state.build_success:
            logger.warning("build fail -> skip extract api")
            state.current_step = "respond"
            return state
        logger.info("=== extract api start ===")
        config = load_build_config()
        lib_config = config[f"{state.lib}_api"]
        lib_path = lib_config.get("lib_dir")
        lib_path = os.path.join("library", state.lib, lib_path)
        logger.info(f"lib_dir: {lib_path}")
        api_list = extract_lib_symbols(lib_path)

        state.api_list = api_list
        if not api_list:
            logger.error("extract api fail: api list is empty")
            state.api_extraction_success = False
            state.current_step = "respond"
            return state
        output_dir = os.path.join('library', state.lib)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'extracted_api.txt')
        with open(output_file, 'w') as f:
            f.write(",".join(api_list))
        logger.info(f"API extraction result saved to {output_file}")
        logger.info(f"API extraction complete: {len(api_list)} APIs")
        state.api_extraction_success = True
        state.current_step = "build_rag_and_fuzz_cov"
        return state
    except Exception as e:
        logger.error(f"extract_api_node exception: {e}")
        state.error = f"[extract_api_node] {e}"
        state.current_step = "respond"
        return state

async def build_rag_and_fuzz_cov_node(state: LibraryAgentState) -> LibraryAgentState:
    try:
        ok_fuzz = True
        ok_cov = True
        if build_check(state.lib_fuzz_dir):
            ok_fuzz = clone_and_build_library(state.lib_fuzz_dir)
            logger.info(f"fuzz build result: {'success' if ok_fuzz else 'fail'}")
        else:
            logger.info(f"fuzz build result: already exist")
        if not ok_fuzz:
            logger.error("=== fuzz build fail ===")
            state.build_success = False
            state.build_log += f"\n=== fuzz build fail ===\n"
            state.current_step = "respond"
            return state
        state.build_success = True
        
        if build_check(state.lib_cov_dir):
            ok_cov = clone_and_build_library(state.lib_cov_dir)
            logger.info(f"cov build result: {'success' if ok_cov else 'fail'}")
        else:
            logger.info(f"cov build result: already exist")
        if not ok_cov:
            logger.error("=== cov build fail ===")
            state.build_success = False
            state.build_log += f"\n=== cov build fail ===\n"
            state.current_step = "respond"
            return state
        
        output_dir = os.path.join('rag_store', state.lib)
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(output_dir,f"{state.lib}_index_rag")) or not os.listdir(os.path.join(output_dir,f"{state.lib}_index_rag")):
            logger.info("=== RAG index build and engine load start ===")
            state.rag_manager.build_library_rag()
            state.rag_manager.load_index_and_engine()
            logger.info("RAG index build and engine load done!")
        else:
            logger.info("=== RAG index exist check ===")
            logger.info("=== engine load start ===")
            state.rag_manager.load_index_and_engine()
            logger.info("engine load done!")
        
        if not os.path.exists(os.path.join(output_dir,f"{state.lib}_prototype")) or not os.listdir(os.path.join(output_dir,f"{state.lib}_prototype")):
            logger.info("=== AST extraction and BM25 embedding start ===")
            config = load_build_config()
            lib_config = config[f"{state.lib}_api"]
            include_dirs = lib_config.get("include_dir") or []
            logger.info(f"[INCLUDE DIRS] {include_dirs}")
            state.rag_manager.build_api_prototype(api_list=state.api_list, include_dirs=include_dirs)
            state.rag_manager.load_bm25()
            logger.info("AST extraction and BM25 embedding done!")
        else:
            logger.info("=== BM25 exist check ===")
            logger.info("=== BM25 load start ===")
            state.rag_manager.load_bm25()
            logger.info("BM25 load done!")
        
        state.ast_chunks_path = os.path.join(output_dir,f"{state.lib}_protorype","chunks.json")

        dashboard_manager.update_coverage_threshold(state.coverage_threshold)
        dashboard_manager.update_run_index(state.current_run_index)
        state.current_step = "categorize_api"
        return state
    except Exception as e:
        logger.error(f"build_rag_node exception: {e}")
        state.error = f"[build_rag_node] {e}"
        state.current_step = "respond"
        return state

async def categorize_api_node(state: LibraryAgentState) -> LibraryAgentState:
    try:
        if not state.api_list:
            logger.error("API list is empty, cannot proceed with categorization.")
            state.categorized_apis = {}
            state.current_step = "respond"
            return state
            
        if state.retry_count >= state.max_retries:
            logger.error(f"Maximum retry count({state.max_retries}) reached. Stopping API categorization.")
            state.error = f"[categorize_api_node] Maximum retry count exceeded"
            state.current_step = "respond"
            return state
            
        logger.info(f"=== API category classification start (using RAG) - attempt {state.retry_count + 1}/{state.max_retries} ===")
        
        if state.current_run_index > 0:
            state.unused_apis = get_unused_apis(state.lib, state.api_list)
        else:
            state.unused_apis = []
        try:
            task = asyncio.create_task(braille.play_braille_frames_with_timer(braille.frames, delay=0.05)())
            async with llm_semaphore:
                rag_response = await state.rag_manager.query_categorize(
                    lib=state.lib, 
                    api_list=state.api_list, 
                    unused_apis=state.unused_apis if state.unused_apis else None
                )
        except Exception as e:
            logger.error(f"query_categorize exception: {e}")
            raise
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        llm_response = clean_llm_response(str(rag_response))
        logger.info(f"RAG original response: {rag_response!r}")
        logger.info(f"RAG cleaned response: {llm_response!r}")
        logger.info(f"RAG response length: {len(llm_response)}")
        categorized_apis = extract_json_from_response(llm_response)
        if not categorized_apis:
            logger.error("API classification failed: Could not extract JSON")
            state.error = f"[categorize_api_node] JSON extraction failed - response: {llm_response[:200]}..."
            state.categorized_apis = {}
            state.retry_count += 1
            state.current_step = "retry_categorize"
            return state
        
        state.categorized_apis = categorized_apis
        if not state.categorized_apis:
            logger.error("API classification failed: Extracted categories are empty")
            state.error = f"[categorize_api_node] categories are empty"
            state.retry_count += 1
            state.current_step = "retry_categorize"
            return state
            
        if len(state.categorized_apis) == 0:
            logger.error("API classification failed: Number of categories is 0")
            state.error = f"[categorize_api_node] Number of categories is 0"
            state.retry_count += 1
            state.current_step = "retry_categorize"
            return state
            
        output_file = os.path.join('library', state.lib, f'categorized_api_{state.current_run_index:03d}.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(state.categorized_apis, f, indent=2, ensure_ascii=False)
        logger.info(f"API classification result saved to {output_file} (categories: {len(state.categorized_apis)})")
        
        state.retry_count = 0
        state.current_step = "make_subprocess"
        return state
    except Exception as e:
        logger.error(f"categorize_api_node exception: {e}")
        state.error = f"[categorize_api_node] {e}"
        state.retry_count += 1
        if state.retry_count >= state.max_retries:
            state.current_step = "respond"
        else:
            state.current_step = "retry_categorize"
        return state

async def generate_ast_node(state: LibraryAgentState) -> LibraryAgentState:
    try:
        logger.info("=== AST generation start ===")
        ast_success, ast_log = generate_ast_files(state.lib_api_dir)
        logger.info(ast_log)
        state.current_step = "build_rag"
        return state
    except Exception as e:
        logger.error(f"generate_ast_node exception: {e}")
        state.error = f"[generate_ast_node] {e}"
        state.current_step = "respond"
        return state

async def make_subprocess_node(state: LibraryAgentState) -> LibraryAgentState:
    """make_subprocess node for parallel harness generation by category"""
    try:
        logger.info("=== make_subprocess node start ===")
        logger.info(f"Number of categories: {len(state.categorized_apis)}")

        if not state.categorized_apis:
            logger.error("No category classification results, skipping harness generation.")
            state.error = f"[make_subprocess_node] No category classification results"
            state.current_step = "retry_categorize"
            return state
            
        if len(state.categorized_apis) == 0:
            logger.error("Number of categories is 0. Skipping harness generation.")
            state.error = f"[make_subprocess_node] Number of categories is 0"
            state.current_step = "retry_categorize"
            return state
        
        constraint_dir = os.path.join("rag_store", state.lib, f"{state.lib}_constraint_rag")
        os.makedirs(constraint_dir, exist_ok=True)

        results = await run_parallel_harness_generation(
            categorized_apis=state.categorized_apis, 
            lib_api_dir=state.lib_api_dir, 
            total_api_list=state.api_list,
            lib=state.lib,
            lib_fuzz_dir=state.lib_fuzz_dir,
            lib_cov_dir=state.lib_cov_dir,
            ast_chunks_path=state.ast_chunks_path,
            rag_manager=state.rag_manager,
            run_index=state.current_run_index,
            unused_apis=state.unused_apis,
            stop_early=state.stop_early
        )
        
        logger.info("=== make_subprocess node completed ===")
        state.current_step = "combine_coverage"
        return state
        
    except Exception as e:
        logger.exception(f"make_subprocess_node exception")
        state.error = f"[make_subprocess_node] {e}"
        state.current_step = "respond"
        return state

async def combine_coverage_node(state: LibraryAgentState) -> LibraryAgentState:
    """Select harnesses based on coverage and calculate total library coverage"""
    try:
        state.total_coverage = filter_and_combine_coverage(
            lib=state.lib,
            categorized_apis=state.categorized_apis,
            current_run_index=state.current_run_index,
            unused_apis=state.unused_apis
        )
        dashboard_manager.update_total_coverage(state.total_coverage)
        logger.info(f"Coverage threshold: {state.coverage_threshold}%")
        
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdin.flush()
        gc.collect()

        if state.total_coverage >= state.coverage_threshold:
            logger.info(f"Coverage goal achieved! ({state.total_coverage:.2f}% >= {state.coverage_threshold}%)")
            logger.info("Harness generation and fuzzing completed successfully!")
            state.current_step = "respond"
        else:
            logger.warning(f"Coverage goal not achieved ({state.total_coverage:.2f}% < {state.coverage_threshold}%)")
            
            if state.max_time is not None:
                elapsed = time.time() - dashboard_manager.start_time
                if elapsed >= state.max_time:
                    logger.warning(f"Maximum execution time ({state.max_time}s) exceeded! Stopping category classification (elapsed: {elapsed:.0f}s)")
                    state.current_step = "respond"
                    return state
            
            if state.current_run_index < state.max_runs:
                state.current_run_index += 1
                dashboard_manager.update_run_index(state.current_run_index)
                logger.info(f"🔁 Next run (run_index={state.current_run_index}/{state.max_runs})")
                state.categorized_apis = {}
                state.retry_count = 0
                state.current_step = "categorize_api"
                await asyncio.sleep(1)
                return state
            else:
                logger.warning("Maximum number of runs reached. Stopping.")
                state.current_step = "respond"
        return state
        
    except Exception as e:
        logger.error(f"combine_coverage_node exception: {e}")
        logger.exception("Detailed exception information:")
        state.error = f"[combine_coverage_node] {e}"
        state.total_coverage = 0.0
        state.current_step = "respond"
        return state

def generate_response(state: LibraryAgentState) -> LibraryAgentState:
    if state.build_success is False:
        content = f"libpng build failed!\n{state.build_log}"
        state.current_step = "end"
    elif not state.api_extraction_success:
        content = f"API extraction failed!\nBuild succeeded but API list could not be extracted."
        state.current_step = "end"
    else:
        api_list = state.api_list
        categorized_apis = state.categorized_apis
        if api_list and categorized_apis:
            content = (
                f"libpng API function count: {len(api_list)}\n\n"
                f"Category-wise classification:\n{json.dumps(categorized_apis, indent=2, ensure_ascii=False)}\n\n"
                f"AST files have been generated:\n"
                f"- library/libpng/ast/: AST(.ast) for all source files\n\n"
            )
            
            if state.total_coverage > 0:
                content += f"Total coverage: {state.total_coverage:.1f}%\n"
                if state.total_coverage >= state.coverage_threshold:
                    content += f"Coverage threshold({state.coverage_threshold}%) achieved! Next step completed.\n"
                else:
                    content += f"Coverage threshold({state.coverage_threshold}%) not achieved. Harness regeneration required.\n"
        elif api_list:
            content = f"libpng API function count: {len(api_list)}\n\nCategory classification failed\n\nAST files have been generated."
        else:
            content = "API extraction failed"
        state.current_step = "end"
    
    state.response = content
    return state

def visualize_workflow(workflow, lib: str):
    """Visualize workflow graph."""
    try:
        workflow_log_dir = os.path.join("log", "workflow")
        os.makedirs(workflow_log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y%m%d_%H%M%S')
        png_filename = f"workflow_{lib}_{timestamp}.png"
        mmd_filename = f"workflow_{lib}_{timestamp}.mmd"
        
        png_path = os.path.join(workflow_log_dir, png_filename)
        mmd_path = os.path.join(workflow_log_dir, mmd_filename)
        
        graph = workflow.get_graph()
        with open(png_path, "wb") as f:
            f.write(graph.draw_mermaid_png())
        logger.info(f"Workflow graph saved to {png_path}")
        
        mermaid_code = workflow.get_graph().draw_mermaid()
        logger.info("Successfully retrieved Mermaid code.")
        
        with open(mmd_path, "w") as f:
            f.write(mermaid_code)
        logger.info(f"Mermaid code saved to {mmd_path}")
        
    except Exception as e:
        logger.warning(f"Failed to visualize graph: {e}")

class LibraryAgent:
    def __init__(self, lib: str):
        if '_' in lib:
            raise ValueError("--lib argument must be a name without purpose, like 'libpng'. (e.g., --lib libpng)")
        
        self.lib = lib
        self.lib_api_dir = f"{lib}_api"
        self.lib_fuzz_dir = f"{lib}_fuzz"
        self.lib_cov_dir = f"{lib}_cov"
        self.log_filename = setup_unified_logger(self.lib)
        global logger
        logger = get_logger(__name__)
        
        # Restore cumulative token counts from previous run if token_usage.json exists
        dashboard_manager.load_token_snapshot(get_token_file())
        if dashboard_manager.token_call_count > 0:
            logger.info(f"[TOKEN] Resumed: IN={dashboard_manager.total_input_tokens:,} OUT={dashboard_manager.total_output_tokens:,} calls={dashboard_manager.token_call_count}")
        
        # Initialize loggers for other modules
        harness_agent.logger = get_logger('harness_agent')
        module.logger = get_logger('module')
        
        # Ensure afk Docker image has all library dependencies
        sync_image_dependencies()
        
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(LibraryAgentState)
        
        # Explicitly set messages channel
        workflow.add_node("process_input", process_user_input)
        workflow.add_node("build_library", build_library_node)
        workflow.add_node("extract_api", extract_api_node)
        workflow.add_node("build_rag_and_fuzz_cov", build_rag_and_fuzz_cov_node)
        workflow.add_node("categorize_api", categorize_api_node)
        workflow.add_node("make_subprocess", make_subprocess_node)
        workflow.add_node("combine_coverage", combine_coverage_node)
        workflow.add_node("respond", generate_response)
        
        workflow.add_edge(START, "process_input")
        workflow.add_edge("process_input", "build_library")
        
        workflow.add_conditional_edges(
            "build_library",
            lambda state: state.current_step,
            {
                "extract_api": "extract_api",
                "respond": "respond"
            }
        )
        workflow.add_conditional_edges(
            "extract_api",
            lambda state: state.current_step,
            {
                "build_rag_and_fuzz_cov": "build_rag_and_fuzz_cov",
                "respond": "respond"
            }
        )
        workflow.add_conditional_edges(
            "build_rag_and_fuzz_cov",
            lambda state: state.current_step,
            {
                "categorize_api": "categorize_api",
                "respond": "respond",
            }
        )
        
        workflow.add_conditional_edges(
            "categorize_api",
            lambda state: state.current_step,
            {
                "make_subprocess": "make_subprocess",
                "retry_categorize": "categorize_api",  
                "respond": "respond"
            }
        )
        
        workflow.add_conditional_edges(
            "make_subprocess",
            lambda state: state.current_step,
            {
                "combine_coverage": "combine_coverage",
                "retry_categorize": "categorize_api",  
                "respond": "respond"
            }
        )
        
        workflow.add_conditional_edges(
            "combine_coverage",
            lambda state: state.current_step,
            {
                "respond": "respond",
                "categorize_api": "categorize_api"
            }
        )
        
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    async def run(self, coverage_threshold: Optional[float] = None, context_window: Optional[int] = None, max_time: Optional[int] = None, stop_early: bool = False) -> dict:
        
        manager = OllamaRAGManager(
            lib=self.lib, 
            model=LLM_MODEL_NAME, 
            base_url=LLM_SERVER_URL, 
            embedding_model_name=EMBEDDING_MODEL_NAME, 
            context_window=context_window,
            token_callback=lambda in_tok, out_tok: dashboard_manager.update_token_usage(in_tok, out_tok, token_file=get_token_file())
        )
        
        initial_state = LibraryAgentState(
            lib=self.lib,
            lib_api_dir=self.lib_api_dir,
            lib_fuzz_dir=self.lib_fuzz_dir,
            lib_cov_dir=self.lib_cov_dir,
            rag_manager=manager,
            coverage_threshold=(coverage_threshold if coverage_threshold is not None else 20.0),
            max_time=max_time,
            stop_early=stop_early
        )
        result = await self.graph.ainvoke(initial_state, {"recursion_limit": float('inf')})
        return {
            "api_list": result['api_list'],
            "build_success": result['build_success'],
            "build_log": result['build_log'],
            "categorized_apis": result['categorized_apis'],
            "response": result['response'],
            "log_filename": self.log_filename
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="Library build and API extraction agent",
        epilog="""
Examples:
  python3 agent.py --lib libpng"
  python3 agent.py -l libpng"
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-l", "--lib", type=str, required=True, help="Library name (based on build_config.yaml)")
    parser.add_argument("-c", "--coverage-threshold", type=float, default=100.0, help="Final coverage threshold (%%)")
    parser.add_argument("-w", "--context-window", type=int, default=60000, help="context window size")
    parser.add_argument("-m", "--max-time", type=int, default=None, help="Maximum execution time in seconds. If exceeded, a new category will not be started (default: no limit, e.g., 86400=24 hours)")
    parser.add_argument("-s", "--stop-early", action="store_true", help="Stop after LLM judge, do not proceed to long fuzzing")
    args = parser.parse_args()

    try:
        agent = LibraryAgent(args.lib)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    result = asyncio.run(agent.run(
        coverage_threshold=args.coverage_threshold, 
        context_window=args.context_window,
        max_time=args.max_time,
        stop_early=args.stop_early
    ))

    logger.info(f"log file: {result['log_filename']}") 