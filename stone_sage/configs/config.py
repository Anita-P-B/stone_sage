class Config:
    def __init__(
        self,
        # dataset
        DATA_PATH= r"./data/concrete_data.csv",
        CHECKSUM="3a8e1fe4362dc3fbff27ef2eaa42c8390ad970cdb6931479876f47dd23ead7d5",  # SHA-256 of the known-good file
        FORCE_DOWNLOAD = False,
        SMALL_DATASET=False,

        # General constants
        BATCH_SIZE=32,
        EPOCHS=100,
        MODEL = None,
        LEARNING_RATE=1e-3,
        SCHEDULER = True, # Set to False to disable learning rate scheduler
        DROPOUT_RATE = 0.5,
        WEIGHT_DECAY = 1e-4,
        AUGMENTATION_PROB=0.3,
        NORM=None,  # None|"mean"

        # Scheduler Config
        MODE = "min",
        FACTOR = 0.7,
        PATIENCE = 10,
        MIN_LR = 1e-6,
        VERBOSE = True,

        # Paths
        SAVE_PATH="test",
        LOAD_MODEL=  None,
        RUN_DIR_BASE="./experiments",
        CHECKPOINT_PATH = None,

        # Data split
        SPLIT_SEED=42,
        VAL_RATIO=0.8, # Ratio of validation set from the full test set

        # Sweep flag
        SWEEP_MODE = False,

        # Debug
        DEBUG = False

    ):
        # dataset
        self.DATA_PATH =DATA_PATH
        self.CHECKSUM = CHECKSUM
        self.FORCE_DOWNLOAD = FORCE_DOWNLOAD
        self.SMALL_DATASET = SMALL_DATASET

        # General constants
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.MODEL = MODEL
        self.LEARNING_RATE = LEARNING_RATE
        self.SCHEDULER = SCHEDULER
        self.DROPOUT_RATE = DROPOUT_RATE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.AUGMENTATION_PROB = AUGMENTATION_PROB
        self.NORM = NORM

        # Scheduler Config
        self.MODE = MODE
        self.FACTOR = FACTOR
        self.PATIENCE = PATIENCE
        self.MIN_LR = MIN_LR
        self.VERBOSE = VERBOSE

        # Paths
        self.SAVE_PATH = SAVE_PATH
        self.LOAD_MODEL = LOAD_MODEL
        self.RUN_DIR_BASE = RUN_DIR_BASE
        self.CHECKPOINT_PATH = CHECKPOINT_PATH

        # Data split
        self.SPLIT_SEED = SPLIT_SEED
        self.VAL_RATIO = VAL_RATIO

        # Sweep mode
        self.SWEEP_MODE = SWEEP_MODE

        #Debug
        self.DEBUG = DEBUG

    def update_from_dict(self, config_dict: dict, verbose=True):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if verbose:
                    print(f"✅ Set {key} = {value}")
            else:
                if verbose:
                    print(f"⚠️ Skipped unknown config key: {key}")

    def to_dict(self):
        return self.__dict__.copy()