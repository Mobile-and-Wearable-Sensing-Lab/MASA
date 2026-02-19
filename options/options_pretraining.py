
class opts_SLR():

    def __init__(self):
        self.num_class = 128

        # feeder
        self.train_feeder_args = {
            'data_path': '/home/nithin/Desktop/ISL_Goa_Data/MASA/Data/ISL_GOA',
            'num_frame_path': '',
            'l_ratio': [0.1, 1],
            'input_size': 64,
            'mask_ratio':None
        }
