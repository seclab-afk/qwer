# harness_agent.py
# Agent for category-based harness generation
import logging
import os
import json
import asyncio
import sys
import gc
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from langgraph.graph import StateGraph, START, END
from module import OllamaRAGManager, should_regenerate_harness, build_fuzzer, run_docker_fuzzing, measure_coverage, check_fuzz_done, clean_llm_response, extract_code_from_response, remove_docker_container, llm_semaphore, cleanup_all_fuzzer_containers, move_successful_harness, save_attempt_log, reproduce_crash, parse_fuzzer_stats, extract_flow_from_response, get_unused_apis
from dashboard_manager import dashboard_manager, HarnessStage
from logger_config import get_logger

# Module-level logger setup (initialized later)
logger = None

class HarnessAgentState(BaseModel):
    """State for harness generation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    category: str = ""
    api_list: List[str] = Field(default_factory=list)
    total_api_list: List[str] = Field(default_factory=list)
    api_flow: List[dict] = Field(default_factory=list)
    lib_api_dir: str = ""
    lib: str = ""
    lib_fuzz_dir: str = ""
    lib_cov_dir: str = ""
    fuzzer_dir: str = ""
    ast_chunks_path: str =""
    harness_code: str = ""
    error: str = ""
    success: bool = False
    build_success: bool = False
    current_step: str = "generate_api_flow"
    coverage_increased: bool = False
    previous_coverage: float = 0.0
    current_coverage: float = 0.0
    fuzz_short_duration: int = 600  # 10 minutes
    fuzz_long_duration: int = 86400  # 24 hours
    is_long_fuzz: bool = False  # Whether currently in long fuzzing stage
    rag_manager: Optional[OllamaRAGManager] = None  # RAG manager (or NoRAG manager)
    build_error: str = ""  # Build error message
    fix_retry_count: int = -1  # fix_harness retry count (starts at -1, becomes 0 after first increment)
    generate_retry_count: int = -1  # generate_harness retry count (starts at -1, becomes 0 after first increment)
    flow_retry_count: int = 0  # flow generation retry count (within current generate attempt)
    max_fix_retries: int = 5  # Maximum fix_harness retries
    max_generate_retries: int = 3  # Maximum generate_harness retries
    max_flow_retries: int = 5  # Maximum flow generation retries
    agent_id: str = ""  # Unique agent ID
    run_index: int = 0  # Run index (for separating multiple run results)
    cpu_no: Optional[str] = None  # Assigned CPU number (for cpuset, string)
    crash_outputs: List[str] = Field(default_factory=list)  # crash reproduce outputs
    llm_judgement_reason: str = ""  # LLM judgement reason
    unused_apis: List[str] = Field(default_factory=list)  # Unused API list (for flow generation reference)
    stop_early: bool = False  # If True, stop after LLM judgement and do not proceed to long fuzzing


async def make_api_flow_node(state: HarnessAgentState) -> HarnessAgentState:
    """API flow generation node before harness generation - max flow generation retries"""
    try:
        # Increment generate_retry_count and check max (must be done here so flow and harness are saved in same directory)
        state.generate_retry_count += 1
        state.fix_retry_count = 0  # Reset fix count for new flow generation
        
        # Check max
        if state.generate_retry_count >= state.max_generate_retries:
            logger.warning(f"Category '{state.category}' reached max generate retries ({state.max_generate_retries}). Terminating.")
            try:
                dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, f"Max generate retries ({state.max_generate_retries}) reached")
            except Exception:
                pass
            state.current_step = "end"
            return state
        
        try:
            dashboard_manager.update_generate_retry_count(state.category, state.generate_retry_count)
            dashboard_manager.update_fix_retry_count(state.category, state.fix_retry_count)
        except Exception:
            pass
        
        dashboard_manager.update_category_stage(state.category, HarnessStage.FLOW, 0.0)
        logger.info(f"=== FLOW '{state.category}' API FLOW generation start (gen{state.generate_retry_count}) ===")
        
        if state.rag_manager is None:
            logger.error(f"FLOW '{state.category}': RAG manager is not initialized.")
            state.error = "RAG manager is not initialized."
            state.success = False
            state.current_step = "end"
            return state
        
        # Flow generation max retries
        api_flow = None
        state.flow_retry_count = 0  # Reset for new flow generation
        for attempt in range(1, state.max_flow_retries + 1):
            state.flow_retry_count = attempt
            logger.info(f"FLOW '{state.category}' LLM call (attempt {attempt}/{state.max_flow_retries})")
            try:
                # Get unused APIs from state (None if first run)
                unused_apis = state.unused_apis if state.unused_apis else None
                async with llm_semaphore:
                    flow_response = await state.rag_manager.query_flow(state.lib, state.category, state.api_list, state.total_api_list, unused_apis=unused_apis)
            except Exception as e:
                logger.error(f"FLOW '{state.category}' RAG query failed (attempt {attempt}): {e}")
                if attempt == state.max_flow_retries:
                    state.error = f"Flow generation failed ({state.max_flow_retries} attempts): {e}"
                    state.success = False
                    state.current_step = "end"
                    return state
                continue
            
            flow_response_clean = clean_llm_response(str(flow_response))
            api_flow = extract_flow_from_response(flow_response_clean)
            
            if api_flow:
                logger.info(f"FLOW '{state.category}' parsing success (attempt {attempt})")
                break
            else:
                logger.warning(f"FLOW '{state.category}' parsing failed (attempt {attempt}): {flow_response_clean[:100]}")
        
        if not api_flow:
            logger.error(f"FLOW '{state.category}' parsing failed after {state.max_flow_retries} attempts")
            state.error = f"API flow parsing failed ({state.max_flow_retries} attempts)"
            state.success = False
            state.current_step = "end"
            return state
            
        # Create category subdirectory and save file (gen{N}_fix{M} structure)
        safe_category = state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        harness_category_dir = os.path.join('library', state.lib, f'harness_{state.run_index:03d}', safe_category)
        gen_dir = f"gen{state.generate_retry_count}_fix{state.fix_retry_count}"
        gen_path = os.path.join(harness_category_dir, gen_dir)
        os.makedirs(gen_path, exist_ok=True)
        flow_path = os.path.join(gen_path, f"{safe_category}_flow.json")

        # Get API info using BM25 per API
        if state.rag_manager is not None and hasattr(state.rag_manager, 'search_bm25'):
            enriched_api_flow = []
            for api_item in api_flow:
                if isinstance(api_item, dict) and 'api' in api_item:
                    api_name = api_item['api']
                    try:
                        results = state.rag_manager.search_bm25(api_name, top_n=1)
                        if results:
                            chunk, score = results[0]
                            # Add return_type and params
                            api_item['return_type'] = chunk.get('return_type', '')
                            api_item['params'] = chunk.get('params', [])
                            logger.info(f"BM25 enriched: {api_name} (score: {score:.4f})")
                    except Exception as e:
                        logger.warning(f"BM25 search failed for {api_name}: {e}")
                enriched_api_flow.append(api_item)
            api_flow = enriched_api_flow
            logger.info(f"FLOW '{state.category}' BM25 info enrichment complete")
        
        # Constraint RAG search (get API usage constraints)
        if state.rag_manager is not None and hasattr(state.rag_manager, 'search_constraint_bm25'):
            try:
                # Extract API name list
                api_names = [item['api'] for item in api_flow if isinstance(item, dict) and 'api' in item]
                
                if api_names:
                    # Constraint search
                    constraint_results = state.rag_manager.search_constraint_bm25(
                        lib=state.lib,
                        api_names=api_names,
                        top_n=5
                    )
                    
                    if constraint_results:
                        # API constraint mapping
                        constraint_map = {}
                        for chunk, score in constraint_results:
                            api = chunk.get('api', '')
                            # Extract all reason fields in reason1, reason2, ... format
                            reasons = []
                            for key, val in chunk.items():
                                if key.startswith('reason') and val:
                                    reasons.append(str(val))
                            if api and reasons:
                                if api not in constraint_map:
                                    constraint_map[api] = []
                                constraint_map[api].extend(reasons)
                        
                        # Add constraints to api_flow
                        for api_item in api_flow:
                            if isinstance(api_item, dict) and 'api' in api_item:
                                api_name = api_item['api']
                                if api_name in constraint_map:
                                    api_item['constraints'] = constraint_map[api_name]
                                    logger.info(f"Constraint added for {api_name}: {len(constraint_map[api_name])} items")
                        
                        logger.info(f"FLOW '{state.category}' Constraint RAG info enrichment complete ({len(constraint_results)} constraints)")
                    else:
                        logger.info(f"FLOW '{state.category}' No data in Constraint RAG (first run)")
            except Exception as e:
                logger.warning(f"Constraint RAG search failed (continuing): {e}")

        with open(flow_path, 'w', encoding='utf-8') as f:
            f.write(str(api_flow))
            
        logger.info(f"FLOW '{state.category}' JSON file saved: {flow_path} (attempt: {gen_dir})")
        
        # Handle ValidationError when assigning api_flow (in case string wasn't parsed as dict)
        try:
            state.api_flow = api_flow
        except ValidationError as ve:
            logger.error(f"FLOW '{state.category}' api_flow validation failed: {ve}")
            # Retry if flow retry count remains
            if state.flow_retry_count < state.max_flow_retries:
                logger.warning(f"FLOW '{state.category}' ValidationError occurred - retrying flow generation")
                state.error = f"api_flow validation failed: {ve}"
                state.success = False
                state.current_step = "generate_api_flow"
                return state
            else:
                state.error = f"api_flow validation failed (after {state.max_flow_retries} attempts): {ve}"
                state.success = False
                state.current_step = "end"
                return state
        
        state.success = True
        state.current_step = "generate_harness"
        dashboard_manager.update_category_stage(state.category, HarnessStage.FLOW, 1.0)
        logger.info(f"=== FLOW '{state.category}' JSON generation complete ===")
        return state
        
    except Exception as e:
        logger.error(f"Category '{state.category}' harness generation failed: {e}")
        dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, str(e))
        state.error = str(e)
        state.success = False
        return state
    
async def generate_harness_node(state: HarnessAgentState) -> HarnessAgentState:
    """Harness generation node - counter already incremented in make_api_flow_node"""
    try:
        dashboard_manager.update_category_stage(state.category, HarnessStage.GENERATING, 0.1)
        logger.info(f"=== Category '{state.category}' harness generation start (gen{state.generate_retry_count}_fix{state.fix_retry_count}) ===")
        
        
        
        logger.info(f"Category '{state.category}' LLM inference start")
        if state.rag_manager is not None:
            logger.info(f"Category '{state.category}' RAG query start")
            
            # RAG query retry logic (max 3 attempts)
            max_query_retries = 3
            harness_response = None
            last_error = None
            
            for query_attempt in range(1, max_query_retries + 1):
                try:
                    async with llm_semaphore:
                        harness_response = await state.rag_manager.query_make(
                            state.lib, state.category, state.api_flow
                        )
                    break  # Exit loop on success
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"harness '{state.category}' RAG query failed (attempt {query_attempt}/{max_query_retries}): {e}")
            
            # If all retries failed
            if harness_response is None:
                error_msg = f"RAG query failed after {max_query_retries} attempts: {last_error}"
                logger.error(f"harness '{state.category}' {error_msg}")
                
                # Record error in attempt.json
                save_attempt_log(
                    library=state.lib,
                    category=state.category,
                    gen_try=state.generate_retry_count,
                    fix_try=state.fix_retry_count,
                    result="rag_query_failed",
                    error=error_msg,
                    run_index=state.run_index
                )
                
                state.error = error_msg
                state.success = False
                state.current_step = "generate_api_flow"
                return state
        else:
            logger.error(f"Category '{state.category}': RAG manager is not initialized.")
            state.error = "RAG manager is not initialized."
            state.success = False
            return state
        logger.info(f"LLM response: {harness_response}")
        harness_response_clean = clean_llm_response(str(harness_response))
        harness_code = extract_code_from_response(harness_response_clean)
        if not harness_code:
            harness_code = harness_response_clean
            
        safe_category = state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        harness_category_dir = os.path.join('library', state.lib, f'harness_{state.run_index:03d}', safe_category)
        gen_dir = f"gen{state.generate_retry_count}_fix{state.fix_retry_count}"
        gen_path = os.path.join(harness_category_dir, gen_dir)
        os.makedirs(gen_path, exist_ok=True)  # Create directory
        harness_path = os.path.join(gen_path, f"{safe_category}_harness.c")
        with open(harness_path, 'w', encoding='utf-8') as f:
            f.write(harness_code)
            
        logger.info(f"Category '{state.category}' harness file saved: {harness_path} (attempt: {gen_dir})")
        state.harness_code = harness_code
        state.success = True
        state.current_step = "afl_build_and_check"
        dashboard_manager.update_category_stage(state.category, HarnessStage.GENERATING, 1.0)
        logger.info(f"=== Category '{state.category}' harness generation complete ===")
        return state
        
    except Exception as e:
        logger.error(f"Category '{state.category}' harness generation failed: {e}")
        dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, str(e))
        state.error = str(e)
        state.success = False
        return state

async def afl_build_and_check_node(state: HarnessAgentState) -> HarnessAgentState:
    """AFL build node (performs actual build)"""
    try:
        dashboard_manager.update_category_stage(state.category, HarnessStage.BUILDING, 0.2)
        logger.info(f"=== Category '{state.category}' AFL build start ===")
        # Generate fuzzer directory name and save to state
        safe_category = state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        fuzzer_dir = f"{state.lib}_{safe_category}_fuzzer"
        state.fuzzer_dir = fuzzer_dir
        # Also save lib_fuzz_dir, lib_cov_dir to state
        state.lib_fuzz_dir = f"{state.lib}_fuzz"
        state.lib_cov_dir = f"{state.lib}_cov"
        # 1. Build fuzzing binary (try first)
        fuzz_ok, fuzz_build_log, _ = await build_fuzzer(
            library=state.lib,  # Use appropriate library name variable
            category=state.category,     # Use appropriate category variable
            config_path="build_config.yaml",
            purpose="fuzz",      # Fuzzing binary
            run_index=state.run_index,
            gen_try=state.generate_retry_count,
            fix_try=state.fix_retry_count,
        )
        
        if not fuzz_ok:
            # Handle error immediately on AFL build failure
            error_msg = f"AFL build failed: {fuzz_build_log[:100]}..."
            logger.error(f"Category '{state.category}' AFL build failed: {fuzz_build_log}")
            
            # Save build error info to state (include AFL build log)
            full_error_msg = error_msg
            if fuzz_build_log:
                full_error_msg += f"\n\nAFL build log:\n{fuzz_build_log}"
            state.build_error = full_error_msg
            
            # Save build failure log
            save_attempt_log(
                library=state.lib,
                category=state.category,
                gen_try=state.generate_retry_count,
                fix_try=state.fix_retry_count,
                result="build_failed",
                error=full_error_msg,
                run_index=state.run_index
            )
            
            dashboard_manager.update_category_stage(state.category, HarnessStage.FIX, 0.0, error_msg)
            state.build_success = False
            
            # Move to fix stage on build failure. Fix threshold is checked at fix node entry
            state.current_step = "fix_harness"
            return state
        
        # 2. Build coverage measurement binary on AFL build success (must succeed)
        cov_ok, cov_build_log, _ = await build_fuzzer(
            library=state.lib,
            category=state.category,
            config_path="build_config.yaml", 
            purpose="cov",       # Coverage measurement binary
            run_index=state.run_index,
            gen_try=state.generate_retry_count,
            fix_try=state.fix_retry_count,
        )
        if cov_ok:
            logger.info(f"=== cov build success ===")
        else:
            logger.error(f"=== cov build failed ===")
            logger.error(cov_build_log)

        logger.info(f"=== reproduce build start ===")
        reproduce_ok, reproduce_build_log, _ = await build_fuzzer(
            library=state.lib,
            category=state.category,
            config_path="build_config.yaml",
            purpose="api",
            run_index=state.run_index,
            gen_try=state.generate_retry_count,
            fix_try=state.fix_retry_count,
        )
        if reproduce_ok:
            logger.info(f"=== reproduce build success ===")
        else:
            logger.error(f"=== reproduce build failed ===")
            logger.error(reproduce_build_log)
        
        logger.info(f"Category '{state.category}' AFL build and coverage binary build success")
        dashboard_manager.update_category_stage(state.category, HarnessStage.BUILDING, 1.0)
        state.build_success = True
        state.build_error = ""  # Clear error info on build success
        state.current_step = "docker_fuzz_short"
        return state
    except Exception as e:
        logger.error(f"Category '{state.category}' AFL build failed: {e}")
        dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, str(e))
        state.build_success = False
        state.build_error = str(e)
        # Move to fix stage on build exception. Fix threshold is checked at fix node entry
        state.current_step = "fix_harness"
        return state

async def fix_harness_node(state: HarnessAgentState) -> HarnessAgentState:
    """Harness fix node - fix harness based on build error"""
    # Check retry limit and handle increment
    if state.fix_retry_count < state.max_fix_retries:
        state.fix_retry_count += 1
        try:
            dashboard_manager.update_fix_retry_count(state.category, state.fix_retry_count)
        except Exception:
            pass
    else:
        logger.warning(f"Category '{state.category}' fix_harness max retries ({state.max_fix_retries}) exceeded. Switching to generate_harness.")
        # Dashboard update: notify regeneration switch and show fix counter reset
        try:
            state.fix_retry_count = 0
            dashboard_manager.update_fix_retry_count(state.category, state.fix_retry_count)
            dashboard_manager.update_category_stage(state.category,HarnessStage.FIX,0.0,"Max fix retries exceeded -> FIX")
        except Exception:
            pass
        state.current_step = "generate_api_flow"  # Go to generate_api_flow to increment counter
        return state
    try:
        # Dashboard status update
        dashboard_manager.update_category_stage(state.category, HarnessStage.FIX, 0.3)
        logger.info(f"=== Category '{state.category}' harness fix start ===")
        ast_dir = os.path.join('library', state.lib, 'ast')
        logger.info(f"Category '{state.category}' harness fix LLM inference start")
        
        if state.rag_manager is not None:
            logger.info(f"Category '{state.category}' harness fix RAG query start")
            
            # RAG query retry logic (max 3 attempts)
            max_query_retries = 3
            harness_response = None
            last_error = None
            
            for query_attempt in range(1, max_query_retries + 1):
                try:
                    async with llm_semaphore:
                        harness_response = await state.rag_manager.query_fix(
                            state.lib, state.category, state.api_list, 
                            state.harness_code, state.build_error
                        )
                    break  # Exit loop on success
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"harness fix '{state.category}' RAG query failed (attempt {query_attempt}/{max_query_retries}): {e}")
                    
            if harness_response is None:
                error_msg = f"RAG query failed after {max_query_retries} attempts: {last_error}"
                logger.error(f"harness fix '{state.category}' {error_msg}")
                
                # Record error in attempt.json
                safe_category = state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                gen_dir = f"gen{state.generate_retry_count}_fix{state.fix_retry_count}"
                save_attempt_log(
                    library=state.lib,
                    category=state.category,
                    gen_try=state.generate_retry_count,
                    fix_try=state.fix_retry_count,
                    result="rag_query_failed",
                    error=error_msg,
                    run_index=state.run_index
                )
                
                state.error = error_msg
                state.success = False
                # Switch to generate_api_flow on retry failure
                state.current_step = "generate_api_flow"
                return state
        else:
            logger.error(f"Category '{state.category}': RAG manager is not initialized.")
            state.error = "RAG manager is not initialized."
            state.success = False
            return state
        logger.info(f"Harness fix LLM response: {harness_response}")
        harness_response_clean = clean_llm_response(str(harness_response))
        harness_code = extract_code_from_response(harness_response_clean)
        if not harness_code:
            harness_code = harness_response_clean
            
        # Create category subdirectory and save file (gen{N}_fix{M} structure)
        safe_category = state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        harness_category_dir = os.path.join('library', state.lib, f'harness_{state.run_index:03d}', safe_category)
        gen_dir = f"gen{state.generate_retry_count}_fix{state.fix_retry_count}"
        gen_path = os.path.join(harness_category_dir, gen_dir)
        os.makedirs(gen_path, exist_ok=True)
        harness_path = os.path.join(gen_path, f"{safe_category}_harness.c")
        with open(harness_path, 'w', encoding='utf-8') as f:
            f.write(harness_code)
            
        logger.info(f"Category '{state.category}' fixed harness file saved: {harness_path} (attempt: {gen_dir})")
        
        state.harness_code = harness_code
        state.success = True
        state.build_error = ""  # Clear build error info
        state.current_step = "afl_build_and_check"
        
        dashboard_manager.update_category_stage(state.category, HarnessStage.FIX, 1.0)
        logger.info(f"=== Category '{state.category}' harness fix complete ===")
        return state
        
    except Exception as e:
        logger.error(f"Category '{state.category}' harness fix failed: {e}")
        dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, str(e))
        state.error = str(e)
        state.success = False
        return state

async def docker_fuzz_short_node(state: HarnessAgentState) -> HarnessAgentState:
    """10-minute Docker fuzzer execution node"""
    try:
        dashboard_manager.update_category_stage(state.category, HarnessStage.FUZZ_SHORT, 0.4)
        dashboard_manager.update_category_fuzz_info(state.category, state.fuzz_short_duration, False)
        logger.info(f"=== Category '{state.category}' 10-minute Docker fuzzer execution start ===")
        fuzzer_name = state.fuzzer_dir if state.fuzzer_dir else f"{state.lib}_{state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}_fuzzer"
        
        # 1. Start Docker fuzzing (CPU pinning: use state.cpu_no)
        success = await run_docker_fuzzing(
            state.lib,
            fuzzer_name,
            state.fuzz_short_duration,
            state.run_index,
            state.cpu_no,
            state.generate_retry_count,
            state.fix_retry_count
        )
        if not success:
            logger.error(f"Category '{state.category}' Docker fuzzer start failed")
            # Save error info to state and move to measure_coverage_short
            state.error = "Docker fuzzer start failed"
            state.current_step = "measure_coverage_short"
            return state
        
        # 2. Wait for fuzzing completion
        logger.info(f"Category '{state.category}' waiting for fuzzing completion... ({state.fuzz_short_duration} seconds)")
        fuzz_done = await check_fuzz_done(fuzzer_name, state.fuzz_short_duration)
        
        if fuzz_done:
            logger.info(f"Category '{state.category}' 10-minute Docker fuzzer execution complete")
            dashboard_manager.update_category_stage(state.category, HarnessStage.FUZZ_SHORT, 1.0)
            state.is_long_fuzz = False
            state.current_step = "measure_coverage_short"
        else:
            logger.error(f"Category '{state.category}' 10-minute Docker fuzzer execution failed")
            # Save error info to state and move to measure_coverage_short
            state.error = "10-minute fuzzing failed"
            state.current_step = "measure_coverage_short"
        return state
    except Exception as e:
        logger.error(f"Category '{state.category}' 10-minute Docker fuzzer execution failed: {e}")
        # Save error info to state and move to measure_coverage_short
        state.error = f"10-minute Docker fuzzer execution exception: {e}"
        state.current_step = "measure_coverage_short"
        return state

async def docker_fuzz_long_node(state: HarnessAgentState) -> HarnessAgentState:
    """24-hour Docker fuzzer execution node"""
    try:
        dashboard_manager.update_category_stage(state.category, HarnessStage.FUZZ_LONG, 0.8)
        dashboard_manager.update_category_fuzz_info(state.category, state.fuzz_long_duration, True)
        logger.info(f"=== Category '{state.category}' 24-hour Docker fuzzer execution start ===")
        fuzzer_name = state.fuzzer_dir if state.fuzzer_dir else f"{state.lib}_{state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}_fuzzer"

        # 1. Start Docker fuzzing (CPU pinning: use state.cpu_no)
        success = await run_docker_fuzzing(
            state.lib,
            fuzzer_name,
            state.fuzz_long_duration,
            state.run_index,
            state.cpu_no,
            state.generate_retry_count,
            state.fix_retry_count
        )
        if not success:
            logger.error(f"Category '{state.category}' Docker fuzzer start failed")
            state.current_step = "respond"
            return state
        
        # 2. Wait for fuzzing completion
        logger.info(f"Category '{state.category}' waiting for fuzzing completion... ({state.fuzz_long_duration} seconds)")
        fuzz_done = await check_fuzz_done(fuzzer_name, state.fuzz_long_duration)
        
        if fuzz_done:
            logger.info(f"Category '{state.category}' 24-hour Docker fuzzer execution complete")
            dashboard_manager.update_category_stage(state.category, HarnessStage.FUZZ_LONG, 1.0)
            state.is_long_fuzz = True
            state.current_step = "measure_coverage_long"
        else:
            logger.error(f"Category '{state.category}' 24-hour Docker fuzzer execution failed")
            dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, "24-hour fuzzing failed")
            state.current_step = "respond"
        return state
    except Exception as e:
        logger.error(f"Category '{state.category}' 24-hour Docker fuzzer execution failed: {e}")
        dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, str(e))
        state.current_step = "respond"
        return state

async def measure_coverage_short_node(state: HarnessAgentState) -> HarnessAgentState:
    """Coverage measurement and harness quality evaluation after short fuzzing + Docker cleanup"""
    try:
        dashboard_manager.update_category_stage(state.category, HarnessStage.MEASURING, 0.6)
        logger.info(f"=== Category '{state.category}' short fuzzing coverage measurement start ===")
        fuzzer_name = state.fuzzer_dir if state.fuzzer_dir else f"{state.lib}_{state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}_fuzzer"
        cov = await measure_coverage(state.lib, fuzzer_name, state.run_index, state.generate_retry_count, state.fix_retry_count)
        
        # AFL metric-based harness quality evaluation
        dashboard_manager.update_category_stage(state.category, HarnessStage.MEASURING, 1.0)
        if cov is not None:
            state.current_coverage = cov
            try:
                dashboard_manager.update_category_coverage(state.category, cov)
            except Exception:
                pass
        
        # Construct AFL output directory path (gen{N}_fix{M} structure)
        safe_category = state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        category_root = os.path.join("library", state.lib, f"harness_{state.run_index:03d}", safe_category)
        gen_dir = f"gen{state.generate_retry_count}_fix{state.fix_retry_count}"
        fuzzer_output_dir = os.path.join(category_root, gen_dir)
        runtime_minutes = state.fuzz_short_duration // 60  # Convert seconds to minutes
        
        logger.info(f"=== Category '{state.category}' AFL metric-based harness evaluation start ===")
        logger.info(f"AFL output directory: {fuzzer_output_dir}")
        logger.info(f"Runtime: {runtime_minutes} minutes")
        
        # Determine harness quality using actual AFL metrics
        needs_regeneration, reason = should_regenerate_harness(fuzzer_output_dir, runtime_minutes, state.category)
        
        if not needs_regeneration:
            # Harness is OK - proceed to LLM judgement
            logger.info(f"Harness quality evaluation: passed - {reason}")
            state.coverage_increased = True
            
            # Save success log
            save_attempt_log(
                library=state.lib,
                category=state.category,
                gen_try=state.generate_retry_count,
                fix_try=state.fix_retry_count,
                result="short_fuzz_passed",
                coverage_score=cov if cov else 0.0,
                run_index=state.run_index
            )
        else:
            # Harness has issues - needs regeneration
            logger.warning(f"Harness quality evaluation: failed - {reason}")
            state.coverage_increased = False
            
            # Save harness evaluation failure log
            save_attempt_log(
                library=state.lib,
                category=state.category,
                gen_try=state.generate_retry_count,
                fix_try=state.fix_retry_count,
                result="evaluation_failed",
                error=reason,
                coverage_score=cov if cov else 0.0,
                run_index=state.run_index
            )
        
        # Determine next step: proceed to LLM judgement on quality pass, regenerate on failure
        if state.coverage_increased:
            logger.info(f"Category '{state.category}' harness quality passed - moving to LLM judgement")
            state.current_step = "judge_harness"
        else:
            # Docker container cleanup (only cleanup here on quality failure)
            logger.info(f"=== Category '{state.category}' Docker container cleanup start ===")
            docker_success = remove_docker_container(fuzzer_name)
            if docker_success:
                logger.info(f"Category '{state.category}' Docker container cleanup complete")
            else:
                logger.warning(f"Category '{state.category}' Docker container cleanup failed (continuing)")
            
            logger.info(f"Category '{state.category}' harness quality failed - FIX harness")
            dashboard_manager.update_category_stage(state.category, HarnessStage.FIX, 0.0, reason)
            state.current_step = "generate_api_flow"
        
        return state
    except Exception as e:
        logger.error(f"Category '{state.category}' harness evaluation exception: {e}")
        dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, str(e))
        state.coverage_increased = False
        state.current_step = "generate_api_flow"
        return state

async def measure_coverage_long_node(state: HarnessAgentState) -> HarnessAgentState:
    """Coverage measurement after long fuzzing + Docker cleanup (final completion)"""
    try:
        dashboard_manager.update_category_stage(state.category, HarnessStage.MEASURING, 0.6)
        logger.info(f"=== Category '{state.category}' long fuzzing coverage measurement start ===")
        fuzzer_name = state.fuzzer_dir if state.fuzzer_dir else f"{state.lib}_{state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}_fuzzer"
        cov = await measure_coverage(state.lib, fuzzer_name, state.run_index, state.generate_retry_count, state.fix_retry_count)
        
        # Long fuzzing completion processing
        dashboard_manager.update_category_stage(state.category, HarnessStage.MEASURING, 1.0)
        if cov is not None:
            logger.info(f"Category '{state.category}' long fuzzing coverage measurement complete")
            try:
                dashboard_manager.update_category_coverage(state.category, cov)
            except Exception:
                pass
        else:
            logger.warning(f"Category '{state.category}' long fuzzing coverage measurement failed (continuing)")
        
        # Docker container cleanup
        dashboard_manager.update_category_stage(state.category, HarnessStage.DOCKER_CLEANUP, 0.9)
        logger.info(f"=== Category '{state.category}' Docker container cleanup start ===")
        # TODO: Uncomment/enable container cleanup
        docker_success = remove_docker_container(fuzzer_name)
        if docker_success:
            logger.info(f"Category '{state.category}' Docker container cleanup complete")
        else:
            logger.warning(f"Category '{state.category}' Docker container cleanup failed (continuing)")
        
        # Long fuzzing complete - set to final completion state
        logger.info(f"Category '{state.category}' long fuzzing complete - final completion")
        dashboard_manager.update_category_stage(state.category, HarnessStage.COMPLETED, 1.0)
        state.current_step = "respond"
        
        return state
    except Exception as e:
        logger.error(f"Category '{state.category}' long fuzzing coverage measurement failed: {e}")
        dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, str(e))
        state.current_step = "respond"
        return state



async def judge_harness_node(state: HarnessAgentState) -> HarnessAgentState:
    """LLM analyzes harness and crashes to determine whether to proceed with long fuzzing"""
    try:
        dashboard_manager.update_category_stage(state.category, HarnessStage.LLM_JUDGEMENT, 0.5)
        logger.info(f"=== Category '{state.category}' LLM judgement start ===")
        
        # 1. Run crash reproduce (parallel processing directly on host)
        safe_category = state.category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        crash_files, crash_outputs = await reproduce_crash(
            state.lib, state.category, state.run_index,
            state.generate_retry_count, state.fix_retry_count
        )
        state.crash_outputs = crash_outputs
        
        logger.info(f"Category '{state.category}' found {len(crash_files)} crashes")
        
        # 2. Get fuzzing statistics
        gen_dir = f"gen{state.generate_retry_count}_fix{state.fix_retry_count}"
        stats_path = os.path.join(
            "library", state.lib, f"harness_{state.run_index:03d}", 
            safe_category, gen_dir, "out", "default", "fuzzer_stats"
        )
        fuzz_stats = parse_fuzzer_stats(stats_path) or {}
        
        # 3. Request LLM judgement (pass only first crash - rest are saved to file)
        crash_for_judge = crash_outputs[:1] if crash_outputs else []
        async with llm_semaphore:
            should_continue, reason = await state.rag_manager.query_judgement(
                state.lib,
                state.category,
                state.harness_code,
                fuzz_stats,
                crash_for_judge
            )
        
        # Convert reason to JSON string if dict (llm_judgement_reason is str type)
        if isinstance(reason, dict):
            state.llm_judgement_reason = json.dumps(reason, ensure_ascii=False)
        else:
            state.llm_judgement_reason = str(reason)
        logger.info(f"Category '{state.category}' LLM judgement result: {should_continue}, reason: {reason}")
        
        # 4. Add LLM judgement result to attempt.json
        judgement_data = {
            "should_continue": should_continue,
            "reason": reason,
            "crash_count": len(crash_files),
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        attempt_path = os.path.join(
            "library", state.lib, f"harness_{state.run_index:03d}",
            safe_category, gen_dir, "attempt.json"
        )

        # 5. Build constraint RAG from LLM judgement result
        constraint_dir = os.path.join("rag_store", state.lib, f"{state.lib}_constraint_rag")
        os.makedirs(constraint_dir, exist_ok=True)
        
        if not should_continue:
            # 1. Save to JSON file (continue append)
            constraint_path = os.path.join(constraint_dir, f"{state.lib}_constraint.json")
            
            # Read existing file
            if os.path.exists(constraint_path):
                try:
                    with open(constraint_path, 'r', encoding='utf-8') as f:
                        constraint_data = json.load(f)
                except:
                    constraint_data = []
            else:
                constraint_data = []
            
            # Add new entry
            new_entry = {
                "reason": reason
            }
            constraint_data.append(new_entry)
            
            # Save to file
            with open(constraint_path, 'w', encoding='utf-8') as f:
                json.dump(constraint_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"LLM judgement result JSON file saved: {constraint_path} (total {len(constraint_data)} items)")
            
            # 2. Add to BM25 index (dynamic update)
            try:
                if state.rag_manager is not None:
                    state.rag_manager.add_constraint_to_bm25(
                        lib=state.lib,
                        reason=reason,
                    )
                    logger.info(f"LLM judgement result BM25 index addition complete")
            except Exception as e:
                logger.warning(f"BM25 index addition failed (continuing): {e}")
            


        try:
            # Read existing attempt.json
            if os.path.exists(attempt_path):
                with open(attempt_path, 'r', encoding='utf-8') as f:
                    attempt_data = json.load(f)
            else:
                attempt_data = {}
            
            # Add llm_judgement field
            attempt_data["llm_judgement"] = judgement_data
            attempt_data["fuzz_stats"] = fuzz_stats
            
            # Save again
            with open(attempt_path, 'w', encoding='utf-8') as f:
                json.dump(attempt_data, f, indent=2, ensure_ascii=False)
            logger.info(f"LLM judgement result added to attempt.json: {attempt_path}")
        except Exception as save_err:
            logger.warning(f"LLM judgement result save failed: {save_err}")
        
        dashboard_manager.update_category_stage(state.category, HarnessStage.LLM_JUDGEMENT, 1.0)
        
        if should_continue:
            # Move successful harness
            move_success = move_successful_harness(
                state.lib, state.category, state.run_index,
                state.generate_retry_count, state.fix_retry_count
            )
            if move_success:
                logger.info(f"Category '{state.category}' successful harness move complete")
            else:
                logger.warning(f"Category '{state.category}' successful harness move failed")
            
            if state.stop_early:
                dashboard_manager.update_category_stage(state.category, HarnessStage.COMPLETED, 1.0, f"LLM judgement: passed (stop-early)")
                state.current_step = "respond"
            else:
                dashboard_manager.update_category_stage(state.category, HarnessStage.COMPLETED, 1.0, f"LLM judgement: passed")
                state.current_step = "docker_fuzz_long"
        else:
            # reason may be dict so convert to str before slicing
            reason_str = json.dumps(reason, ensure_ascii=False) if isinstance(reason, dict) else str(reason)
            dashboard_manager.update_category_stage(
                state.category, HarnessStage.FIX, 0.0, f"LLM judgement: rejected - {reason_str[:50]}"
            )
            state.current_step = "generate_api_flow"
        # 6. Docker container cleanup (after LLM judgement)
        fuzzer_name = state.fuzzer_dir if state.fuzzer_dir else f"{state.lib}_{safe_category}_fuzzer"
        logger.info(f"=== Category '{state.category}' Docker container cleanup start ===")
        docker_success = remove_docker_container(fuzzer_name)
        if docker_success:
            logger.info(f"Category '{state.category}' Docker container cleanup complete")
        else:
            logger.warning(f"Category '{state.category}' Docker container cleanup failed (continuing)")
        
        return state
        
    except Exception as e:
        logger.error(f"Category '{state.category}' LLM judgement failed: {e}")
        dashboard_manager.update_category_stage(state.category, HarnessStage.FAILED, 0.0, str(e))
        # Default to proceeding with long fuzzing on exception
        state.current_step = "docker_fuzz_long"
        return state

def create_harness_graph(category: str, api_list: List[str], lib_api_dir: str, lib: str, lib_fuzz_dir: str = "", lib_cov_dir: str = "") -> StateGraph:
    """Category-specific harness generation graph"""
    workflow = StateGraph(HarnessAgentState)
    
    workflow.add_node("generate_api_flow", make_api_flow_node)
    workflow.add_node("generate_harness", generate_harness_node)
    workflow.add_node("afl_build_and_check", afl_build_and_check_node)
    workflow.add_node("fix_harness", fix_harness_node)
    workflow.add_node("docker_fuzz_short", docker_fuzz_short_node)
    workflow.add_node("docker_fuzz_long", docker_fuzz_long_node)
    workflow.add_node("measure_coverage_short", measure_coverage_short_node)
    workflow.add_node("measure_coverage_long", measure_coverage_long_node)
    workflow.add_node("judge_harness", judge_harness_node)
    
    workflow.add_edge(START, "generate_api_flow")
    workflow.add_conditional_edges(
        "generate_api_flow",
        lambda state: state.current_step,
        {
            "generate_harness": "generate_harness",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "generate_harness",
        lambda state: state.current_step,
        {
            "afl_build_and_check": "afl_build_and_check",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "fix_harness",
        lambda state: state.current_step,
        {
            "afl_build_and_check": "afl_build_and_check",
            "generate_api_flow": "generate_api_flow", 
            "end": END,
        },
    )
    workflow.add_edge("docker_fuzz_short", "measure_coverage_short")

    workflow.add_conditional_edges(
        "afl_build_and_check",
        lambda state: state.current_step,
        {
            "docker_fuzz_short": "docker_fuzz_short",
            "fix_harness": "fix_harness",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "measure_coverage_short",
        lambda state: state.current_step,
        {
            "judge_harness": "judge_harness",
            "generate_api_flow": "generate_api_flow"
        }
    )
    
    workflow.add_conditional_edges(
        "judge_harness",
        lambda state: state.current_step,
        {
            "docker_fuzz_long": "docker_fuzz_long", 
            "generate_api_flow": "generate_api_flow",
            "respond": END
        }
    )
    
    # Conditional edge: branch based on long fuzzing result  
    workflow.add_conditional_edges(
        "docker_fuzz_long",
        lambda state: state.current_step,
        {
            "measure_coverage_long": "measure_coverage_long",
            "respond": END
        }
    )
    
    workflow.add_edge("measure_coverage_long", END)
    return workflow.compile()

async def run_parallel_harness_generation(categorized_apis: Dict[str, List[str]], lib_api_dir: str, total_api_list: list, lib: str, lib_fuzz_dir: str = "", lib_cov_dir: str = "",ast_chunks_path: str="", rag_manager=None, run_index: int = 0, unused_apis: list = None, stop_early: bool = False) -> Dict[str, str]:
    """Category-based parallel harness generation"""
    try:
        logger.info("=== Parallel harness generation start ===")
        logger.info(f"Number of categories: {len(categorized_apis)}")
        
        dashboard_manager.initialize_categories(categorized_apis, lib, run_index=run_index)
        
        dashboard_task = asyncio.create_task(dashboard_manager.run_dashboard_updater())
        
        await asyncio.sleep(1)
        
        if not categorized_apis:
            logger.warning("No category classification results, skipping harness generation.")
            return {}
        
        if unused_apis:
            logger.info(f"Using unused API feedback: {len(unused_apis)} APIs")
        
        tasks = []
        total_logical = os.cpu_count() or 1
        start_cpu = total_logical // 3
        current_cpu = start_cpu
        for category, api_list in categorized_apis.items():
            logger.info(f"Category '{category}' creating subgraph ({len(api_list)} APIs)")
            
            subgraph = create_harness_graph(category, api_list, lib_api_dir, lib, lib_fuzz_dir, lib_cov_dir)
            
            initial_state = HarnessAgentState(
                category=category,
                api_list=api_list,
                total_api_list= total_api_list,
                lib_api_dir=lib_api_dir,
                lib=lib,
                lib_fuzz_dir=lib_fuzz_dir,
                lib_cov_dir=lib_cov_dir,
                ast_chunks_path = ast_chunks_path,
                fuzz_short_duration=600,
                fuzz_long_duration=86400,
                rag_manager=rag_manager, 
                agent_id=f"{lib}_{category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}", 
                run_index=run_index,
                cpu_no=str(current_cpu),
                unused_apis=unused_apis or [],
                stop_early=stop_early,
            )
            current_cpu = (current_cpu + 1) % total_logical
            
            await asyncio.sleep(0.1) 
            task = asyncio.create_task(subgraph.ainvoke(initial_state, {"recursion_limit": float('inf')}))
            tasks.append((category, task))
        
        logger.info("All category subgraphs running...")
        results = {}
        for category, task in tasks:
            try:
                result = await task
                results[category] = result['harness_code']
                logger.info(f"Category '{category}' complete")
            except Exception as e:
                logger.error(f"Category '{category}' failed: {e}")
                dashboard_manager.update_category_stage(category, HarnessStage.FAILED, 0.0, str(e))
                results[category] = f"Error: {e}"
        
        dashboard_task.cancel()
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass
        
        
        try:
            dashboard_manager.reset_categories()
        except Exception:
            pass
        
        await asyncio.sleep(2)
        await dashboard_manager.stop_dashboard()
        
        logger.info("=== Parallel harness generation complete ===")
        
        logger.info(f"=== Final cleanup of all {lib} containers start ===")
        cleanup_result = cleanup_all_fuzzer_containers(lib)
        logger.info(f"All containers final cleanup complete: {cleanup_result}")
        
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdin.flush()
        gc.collect()
        return results
    except Exception as e:
        logger.exception(f"Exception during parallel harness generation")
        try:
            await dashboard_manager.stop_dashboard()
        except:
            pass
        
        try:
            logger.info(f"=== Cleanup {lib} containers on exception start ===")
            cleanup_result = cleanup_all_fuzzer_containers(lib)
            logger.info(f"Container cleanup on exception complete: {cleanup_result}")
        except Exception as cleanup_error:
            logger.error(f"Container cleanup on exception failed: {cleanup_error}")
        
        return {"error": str(e)}