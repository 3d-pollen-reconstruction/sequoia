import os

class EvalRunner:
    def __init__(self, checkpoints_root, data_dir, output_dir):
        self.checkpoints_root = checkpoints_root
        self.data_dir = data_dir
        self.output_dir = output_dir

    def run(self):
        from eval_pipeline import EvalPipeline
        # Iterate over all models and their checkpoints
        for model_name in os.listdir(self.checkpoints_root):
            model_dir = os.path.join(self.checkpoints_root, model_name)
            if not os.path.isdir(model_dir):
                continue
            for checkpoint_name in os.listdir(model_dir):
                checkpoint_dir = os.path.join(model_dir, checkpoint_name)
                if not os.path.isdir(checkpoint_dir):
                    continue
                print(f"Evaluating model: {model_name}, checkpoint: {checkpoint_name}")
                pipeline = EvalPipeline(
                    checkpoint_dir=checkpoint_dir,
                    data_dir=self.data_dir,
                    output_dir=os.path.join(self.output_dir, model_name, checkpoint_name),
                    model_name=model_name,
                    checkpoint_name=checkpoint_name
                )
                pipeline.run()
        print("Evaluation complete for all models and checkpoints.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation on all model checkpoints.")
    parser.add_argument("--checkpoints_root", default=r"Eval\checkpoints", type=str, help="Root directory containing model checkpoints.")
    parser.add_argument("--data_dir", default="pollen", type=str, help="Directory containing evaluation data.")
    parser.add_argument("--output_dir", default="results", type=str, help="Directory to save evaluation results.")

    args = parser.parse_args()

    runner = EvalRunner(
        checkpoints_root=args.checkpoints_root,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    runner.run()

if __name__ == "__main__":
    main()