class Config:
    def __init__(
        self,
        # dataset
        DATA_PATH= r"./data/concrete_data.csv",
        CHECKSUM="3a8e1fe4362dc3fbff27ef2eaa42c8390ad970cdb6931479876f47dd23ead7d5",  # SHA-256 of the known-good file
        FORCE_DOWNLOAD = False,
        PLOT_STATISTICS=False,
        TARGET_COLUMN = "Concrete compressive strength",
        SHUFFLE = True,

        # Data split
        SPLIT_SEED=42,
        VAL_RATIO=0.1,
        TEST_RATIO = 0.1,

        # General constants
        BATCH_SIZE=16,
        EPOCHS=200,
        MODEL = "shallow_deep" , # set different options in the future
        N_BEST_CHECKPOINTS = 3, # number of best checkpoints to save
        LOSS = "mae",
        OPTIMIZER = "adam" ,
        HIDDEN_DIMS =  [128, 64, 32],
        LEARNING_RATE=0.0005,
        SCHEDULER = True, # Set to False to disable learning rate scheduler
        DROPOUT = 0,

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

        # Sweep flag
        SWEEP_MODE = False,

        # Debug
        DEBUG = False,
        OVERFIT_TEST = False




    ):
        # dataset
        self.DATA_PATH =DATA_PATH
        self.CHECKSUM = CHECKSUM
        self.FORCE_DOWNLOAD = FORCE_DOWNLOAD
        self.PLOT_STATISTICS = PLOT_STATISTICS
        self. TARGET_COLUMN =  TARGET_COLUMN
        self.SHUFFLE  = SHUFFLE

        # Data split
        self.SPLIT_SEED = SPLIT_SEED
        self.VAL_RATIO = VAL_RATIO
        self.TEST_RATIO = TEST_RATIO

        # General constants
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.MODEL = MODEL
        self.N_BEST_CHECKPOINTS =  N_BEST_CHECKPOINTS
        self.LOSS = LOSS
        self.OPTIMIZER = OPTIMIZER
        self.HIDDEN_DIMS =  HIDDEN_DIMS
        self.LEARNING_RATE = LEARNING_RATE
        self.SCHEDULER = SCHEDULER
        self.DROPOUT = DROPOUT

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


        # Sweep mode
        self.SWEEP_MODE = SWEEP_MODE

        #Debug
        self.DEBUG = DEBUG
        self.OVERFIT_TEST = OVERFIT_TEST

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