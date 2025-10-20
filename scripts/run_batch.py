#!/usr/bin/env python3
# Batch processor execution script
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tools.batch_processor.main import main

if __name__ == '__main__':
    sys.exit(main())
