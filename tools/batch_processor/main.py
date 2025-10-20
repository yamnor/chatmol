# Batch processor main entry point
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.batch_processor.config import BatchConfig
from tools.batch_processor.processor import BatchProcessor

def setup_logging(config: BatchConfig):
    """Setup logging configuration."""
    log_level = getattr(logging, config.get('logging.log_level', 'INFO').upper())
    log_file = config.get('logging.log_file', 'data/output/logs/batch_execution.log')
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() if config.get('logging.console_output', True) else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_file}")
    return logger

def load_queries(input_file: str) -> List[str]:
    """Load queries from input file."""
    queries = []
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    queries = data
                elif isinstance(data, dict) and 'queries' in data:
                    queries = data['queries']
                else:
                    raise ValueError("Invalid JSON format")
        else:
            # Assume text file with one query per line
            with open(input_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(queries)} queries from {input_file}")
        return queries
        
    except Exception as e:
        raise Exception(f"Error loading queries from {input_file}: {e}")

def save_results(results: Dict[str, Any], output_dir: str):
    """Save processing results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Batch Processing Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Session ID: {results['session_id']}\n")
        f.write(f"Start Time: {results['start_time']}\n")
        f.write(f"End Time: {results['end_time']}\n")
        f.write(f"Total Duration: {results['total_duration_seconds']:.2f} seconds\n")
        f.write(f"Total Iterations: {results['total_iterations']}\n")
        f.write(f"Successful Iterations: {results['successful_iterations']}\n")
        f.write(f"Failed Iterations: {results['failed_iterations']}\n")
        f.write(f"Success Rate: {(results['successful_iterations']/results['total_iterations']*100):.1f}%\n\n")
        
        if results['errors']:
            f.write(f"Errors:\n")
            for error in results['errors']:
                f.write(f"  Iteration {error['iteration']}: {error['error_type']} - {error['error_message']}\n")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Results saved to: {detailed_file}")
    logger.info(f"Summary saved to: {summary_file}")

def main():
    """Main entry point for batch processor."""
    parser = argparse.ArgumentParser(description='ChatMOL Batch Processor')
    parser.add_argument('--config', '-c', default='config/batch_processor.json',
                       help='Configuration file path')
    parser.add_argument('--input', '-i', default='data/input/queries.txt',
                       help='Input queries file path')
    parser.add_argument('--output', '-o', default='data/output/results',
                       help='Output directory path')
    parser.add_argument('--api-key', help='Gemini API key (overrides config)')
    parser.add_argument('--max-iterations', type=int, help='Maximum iterations (overrides config)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = BatchConfig(args.config)
        
        # Override with command line arguments
        if args.api_key:
            config._set_nested_value(['api', 'key'], args.api_key)
        if args.max_iterations:
            config._set_nested_value(['execution', 'max_iterations'], args.max_iterations)
        
        # Validate configuration
        if not config.validate():
            print("Configuration validation failed")
            return 1
        
        # Setup logging
        logger = setup_logging(config)
        
        # Load queries
        queries = load_queries(args.input)
        if not queries:
            print("No queries found in input file")
            return 1
        
        # Initialize processor
        processor = BatchProcessor(config)
        
        # Run batch processing
        logger.info("Starting batch processing...")
        results = processor.run_batch_processing(queries)
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        print(f"\nBatch Processing Complete!")
        print(f"Session ID: {results['session_id']}")
        print(f"Total Iterations: {results['total_iterations']}")
        print(f"Successful: {results['successful_iterations']}")
        print(f"Failed: {results['failed_iterations']}")
        print(f"Success Rate: {(results['successful_iterations']/results['total_iterations']*100):.1f}%")
        print(f"Duration: {results['total_duration_seconds']:.2f} seconds")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
