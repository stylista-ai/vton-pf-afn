from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument(
            "--warp_checkpoint",
            type=str,
            default="checkpoints/PFAFN/warp_model_final.pth",
            help="load the pretrained model from the specified location",
        )
        self.parser.add_argument(
            "--gen_checkpoint",
            type=str,
            default="checkpoints/PFAFN/gen_model_final.pth",
            help="load the pretrained model from the specified location",
        )
        self.parser.add_argument(
            "--phase", type=str, default="test", help="train, val, test, etc"
        )
        self.parser.add_argument(
            "--unpaired",
            action="store_true",
            help="if enables, uses unpaired data from dataset",
        )
        self.parser.add_argument(
            "--outdir", type=str, help="root output folder for the warp files"
        )
        self.isTrain = False
