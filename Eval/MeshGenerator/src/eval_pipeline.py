class EvalPipeline:
    def __init__(self, checkpoint_dir, data_dir, output_dir, model_name=None, checkpoint_name=None):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.checkpoint_name = checkpoint_name
        self.checkpoints = self.load_checkpoints()

    def load_checkpoints(self):
        import os
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pth') or filename.endswith('.ckpt'):
                checkpoints.append(os.path.join(self.checkpoint_dir, filename))
        return checkpoints

    def evaluate_checkpoints(self):
        for checkpoint in self.checkpoints:
            self.evaluate_mesh(checkpoint)

    def evaluate_mesh(self, checkpoint):
        if self.model_name and self.model_name.lower() == "pixelnerf":
            import sys
            import os
            pixelnerf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pixelnerf.py"))
            if pixelnerf_path not in sys.modules:
                import importlib.util
                spec = importlib.util.spec_from_file_location("pixelnerf", pixelnerf_path)
                pixelnerf = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pixelnerf)
            else:
                pixelnerf = sys.modules["pixelnerf"]

            print(f"Running PixelNeRF evaluation for checkpoint: {checkpoint}")
            runner = pixelnerf.EvalMetricsRunner(
                checkpoint_path=checkpoint,
                output_dir=self.output_dir
            )
            runner.run()
        else:
            print(f"Evaluating mesh for model: {self.model_name}, checkpoint group: {self.checkpoint_name}, checkpoint file: {checkpoint}")

    def run(self):
        self.evaluate_checkpoints()