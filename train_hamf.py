# train_hamf.py
import argparse
import logging
from models.experiment import HAMFExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    try:
        # Initialize experiment
        experiment = HAMFExperiment(args.config)
        
        # Setup components
        experiment.setup()
        
        # Run training
        experiment.train()
        
        # Run evaluation
        results = experiment.evaluate()
        
        logger.info(f"Final evaluation results: {results}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)