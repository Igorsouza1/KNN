const treinox = [
    [1.33,1.31,0.32,1.33,0.46,1.6,1.3,0.45,0.984962406,0.977443609,0.34351145,1.230769231,1  ],
    [1.43,1.37,0.27,1.42,0.51,1.66,1.41,0.42,0.964788732,0.986013986,0.306569343,1.177304965,1  ],
    [1.53,1.36,0.42,1.44,0.58,1.76,1.44,0.51,0.944444444,0.941176471,0.375,1.222222222,0  ],
    [1.34,1.36,0.34,1.37,0.43,1.57,1.36,0.44,0.99270073,1.014925373,0.323529412,1.154411765,0  ],
    [1.37,1.33,0.3,1.37,0.46,1.54,1.35,0.41,0.97080292,0.98540146,0.308270677,1.140740741,1  ],
    [1.42,1.36,0.27,1.4,0.45,1.55,1.38,0.4,0.971428571,0.971830986,0.294117647,1.123188406,0  ],
    [1.46,1.36,0.35,1.47,0.58,1.66,1.48,0.46,0.925170068,1.01369863,0.338235294,1.121621622,0  ],
    [1.45,1.4,0.34,1.45,0.56,1.66,1.44,0.47,0.965517241,0.993103448,0.335714286,1.152777778,0  ],
    [1.41,1.38,0.38,1.44,0.54,1.59,1.41,0.49,0.958333333,1,0.355072464,1.127659574,0  ],
    [1.36,0.51,0.33,1.4,0.47,1.63,1.39,0.45,0.364285714,1.022058824,0.882352941,1.172661871,1  ],
    [1.42,1.34,0.27,1.36,0.45,1.65,1.37,0.4,0.985294118,0.964788732,0.298507463,1.204379562,0  ],
    [1.5,1.42,0.36,1.5,0.55,1.69,1.49,0.45,0.946666667,0.993333333,0.316901408,1.134228188,0  ],
    [1.4,1.33,0.3,1.36,0.43,1.62,1.35,0.41,0.977941176,0.964285714,0.308270677,1.2,0  ],
    [1.43,1.36,0.34,1.4,0.46,1.74,1.42,0.46,0.971428571,0.993006993,0.338235294,1.225352113,0  ],
    [1.46,1.4,0.36,1.43,0.47,1.67,1.3,0.42,0.979020979,0.890410959,0.3,1.284615385,1  ],
    [1.39,1.34,0.28,1.37,0.43,1.5,1.36,0.42,0.97810219,0.978417266,0.313432836,1.102941176,0  ],
    [1.43,1.36,0.38,1.41,0.55,1.68,1.41,0.44,0.964539007,0.986013986,0.323529412,1.191489362,1  ],
    [1.43,1.35,0.37,1.4,0.53,1.69,1.41,0.44,0.964285714,0.986013986,0.325925926,1.19858156,1  ],
    [1.4,1.32,0.34,1.36,0.45,1.62,1.35,0.45,0.970588235,0.964285714,0.340909091,1.2,0  ],
    [1.42,1.32,0.35,1.39,0.55,1.63,1.37,0.44,0.949640288,0.964788732,0.333333333,1.189781022,1  ],
    [1.37,1.35,0.3,1.39,0.46,1.63,1.37,0.42,0.971223022,1,0.311111111,1.189781022,0  ],
    [1.43,1.32,0.37,1.4,0.59,1.69,1.38,0.48,0.942857143,0.965034965,0.363636364,1.224637681,1  ],
    [1.47,1.34,0.38,1.42,0.6,1.69,1.41,0.46,0.943661972,0.959183673,0.343283582,1.19858156,0  ],
    [1.48,1.38,0.38,1.47,0.61,1.61,1.45,0.51,0.93877551,0.97972973,0.369565217,1.110344828,1  ],
    [1.45,1.4,0.34,1.45,0.53,1.62,1.43,0.4,0.965517241,0.986206897,0.285714286,1.132867133,0  ],
    [1.38,1.34,0.27,1.37,0.44,1.77,1.37,0.42,0.97810219,0.992753623,0.313432836,1.291970803,0  ],
    [1.4,1.41,0.34,1.39,0.47,1.62,1.39,0.4,1.014388489,0.992857143,0.283687943,1.165467626,0  ],
    [1.45,1.38,0.34,1.43,0.46,1.61,1.33,0.44,0.965034965,0.917241379,0.31884058,1.210526316,1  ],
    [1.44,1.41,0.34,1.44,0.58,1.61,1.43,0.43,0.979166667,0.993055556,0.304964539,1.125874126,0  ],
    [1.47,1.35,0.36,1.42,0.55,1.66,1.43,0.49,0.950704225,0.972789116,0.362962963,1.160839161,1  ],
    [1.46,1.29,0.35,1.39,0.5,1.64,1.36,0.46,0.928057554,0.931506849,0.356589147,1.205882353,1  ],
    [1.43,1.37,0.36,1.43,0.54,1.68,1.41,0.47,0.958041958,0.986013986,0.343065693,1.191489362,1  ],
    [1.45,1.38,0.31,1.39,0.46,1.63,1.37,0.41,0.992805755,0.944827586,0.297101449,1.189781022,0  ],
    [1.4,1.32,0.34,1.36,0.45,1.62,1.35,0.45,0.970588235,0.964285714,0.340909091,1.2,0  ],
    [1.48,1.38,0.34,1.44,0.51,1.57,1.42,0.46,0.958333333,0.959459459,0.333333333,1.105633803,1  ],
    [1.42,1.34,0.37,1.4,0.56,1.58,1.39,0.44,0.957142857,0.978873239,0.328358209,1.136690647,1  ],
    [1.45,1.4,0.34,1.44,0.53,1.7,1.44,0.42,0.972222222,0.993103448,0.3,1.180555556,0  ],
    [1.41,1.39,0.32,1.42,0.54,1.64,1.4,0.41,0.978873239,0.992907801,0.294964029,1.171428571,1  ],
    [1.39,1.35,0.34,1.38,0.54,1.62,1.37,0.42,0.97826087,0.985611511,0.311111111,1.182481752,1  ],
    [1.4,1.35,0.3,1.39,0.45,1.66,1.31,0.43,0.971223022,0.935714286,0.318518519,1.267175573,1  ],
    [1.39,1.31,0.25,1.37,0.46,1.58,1.36,0.4,0.95620438,0.978417266,0.305343511,1.161764706,0  ],
    [1.4,1.34,0.32,1.4,0.56,1.65,1.37,0.41,0.957142857,0.978571429,0.305970149,1.204379562,1  ],
    [1.47,1.43,0.32,1.47,0.56,1.6,1.45,0.41,0.972789116,0.986394558,0.286713287,1.103448276,0  ],
    [1.47,1.44,0.36,1.47,0.45,1.61,1.33,0.47,0.979591837,0.904761905,0.326388889,1.210526316,1  ],
    [1.42,1.38,0.33,1.41,0.53,1.73,1.39,0.4,0.978723404,0.978873239,0.289855072,1.244604317,1  ],
    [1.31,1.33,0.28,1.36,0.45,1.61,1.35,0.4,0.977941176,1.030534351,0.30075188,1.192592593,0  ],
    [1.45,1.42,0.33,1.45,0.55,1.86,1.43,0.42,0.979310345,0.986206897,0.295774648,1.300699301,1  ],
    [1.42,1.35,0.3,1.39,0.44,1.6,1.37,0.43,0.971223022,0.964788732,0.318518519,1.167883212,0  ],
    [1.36,1.35,0.28,1.38,0.45,1.6,1.36,0.44,0.97826087,1,0.325925926,1.176470588,0  ],
    [1.42,1.38,0.36,1.44,0.54,1.7,1.42,0.44,0.958333333,1,0.31884058,1.197183099,0  ],
    [1.41,1.37,0.32,1.4,0.45,1.64,1.39,0.46,0.978571429,0.985815603,0.335766423,1.179856115,0  ],
    [1.44,1.36,0.38,1.42,0.55,1.69,1.42,0.44,0.957746479,0.986111111,0.323529412,1.190140845,1  ],
    [1.41,1.33,0.3,1.35,0.44,1.69,1.36,0.4,0.985185185,0.964539007,0.30075188,1.242647059,0  ],
    [1.39,1.31,0.26,1.35,0.46,1.6,1.35,0.42,0.97037037,0.971223022,0.320610687,1.185185185,0  ],
    [1.37,1.37,0.27,1.38,0.45,1.5,1.37,0.38,0.992753623,1,0.277372263,1.094890511,0  ],
    [1.51,1.36,0.38,1.45,0.62,1.64,1.43,0.46,0.937931034,0.947019868,0.338235294,1.146853147,1  ],
    [1.44,1.35,0.31,1.41,0.55,1.65,1.4,0.45,0.957446809,0.972222222,0.333333333,1.178571429,1  ],
    [1.41,1.41,0.3,1.45,0.49,1.68,1.45,0.42,0.972413793,1.028368794,0.29787234,1.15862069,0  ],
    [1.39,1.31,0.31,1.39,0.54,1.46,1.36,0.42,0.942446043,0.978417266,0.320610687,1.073529412,1  ],
    [1.33,1.31,0.32,1.33,0.46,1.6,1.3,0.45,0.984962406,0.977443609,0.34351145,1.230769231,0  ],
    [1.41,1.36,0.32,1.4,0.54,1.59,1.38,0.42,0.971428571,0.978723404,0.308823529,1.152173913,1  ],
    [1.43,1.37,0.31,1.42,0.52,1.72,1.41,0.45,0.964788732,0.986013986,0.328467153,1.219858156,1  ],
    [1.48,1.41,0.36,1.42,0.58,1.63,1.42,0.45,0.992957746,0.959459459,0.319148936,1.147887324,0  ],
    [1.48,1.31,0.36,1.39,0.58,1.59,1.37,0.45,0.942446043,0.925675676,0.34351145,1.160583942,1  ],
    [1.43,1.4,0.32,1.44,0.56,1.65,1.41,0.43,0.972222222,0.986013986,0.307142857,1.170212766,1  ],
    [1.35,1.33,0.33,1.34,0.44,1.48,1.33,0.42,0.992537313,0.985185185,0.315789474,1.112781955,0  ],
    [1.44,1.34,0.38,1.43,0.69,1.6,1.4,0.48,0.937062937,0.972222222,0.358208955,1.142857143,0  ],
    [1.4,1.31,0.33,1.41,0.56,1.58,1.37,0.46,0.929078014,0.978571429,0.351145038,1.153284672,1  ],
    [1.43,1.35,0.32,1.41,0.54,1.58,1.42,0.42,0.957446809,0.993006993,0.311111111,1.112676056,0  ],
    [1.45,1.38,0.39,1.45,0.48,1.61,1.43,0.48,0.951724138,0.986206897,0.347826087,1.125874126,0  ],
    [1.41,1.38,0.28,1.39,0.44,1.6,1.38,0.4,0.992805755,0.978723404,0.289855072,1.15942029,0  ],
    [1.44,1.42,0.37,1.44,0.49,1.64,1.32,0.46,0.986111111,0.916666667,0.323943662,1.242424242,1  ],
    [1.41,1.36,0.36,1.4,0.56,1.64,1.38,0.4,0.971428571,0.978723404,0.294117647,1.188405797,1  ],
    [1.45,1.4,0.35,1.45,0.57,1.67,1.43,0.41,0.965517241,0.986206897,0.292857143,1.167832168,0  ],
    [1.38,1.33,0.26,1.37,0.43,1.61,1.35,0.42,0.97080292,0.97826087,0.315789474,1.192592593,0  ],
    [1.48,1.42,0.36,1.49,0.49,1.69,1.49,0.47,0.953020134,1.006756757,0.330985915,1.134228188,0  ],
    [1.38,1.35,0.29,1.38,0.45,1.66,1.36,0.4,0.97826087,0.985507246,0.296296296,1.220588235,0  ],
    [1.33,1.29,0.32,1.35,0.48,1.51,1.31,0.42,0.955555556,0.984962406,0.325581395,1.152671756,0  ],
    [1.41,1.41,0.3,1.45,0.49,1.68,1.45,0.42,0.972413793,1.028368794,0.29787234,1.15862069,0  ],
    [1.43,1.31,0.3,1.35,0.45,1.62,1.35,0.41,0.97037037,0.944055944,0.312977099,1.2,0  ],
    [1.42,1.36,0.29,1.39,0.48,1.69,1.4,0.4,0.978417266,0.985915493,0.294117647,1.207142857,0  ],
    [1.3,1.32,0.29,1.35,0.45,1.66,1.35,0.42,0.977777778,1.038461538,0.318181818,1.22962963,0  ],
    [1.44,1.34,0.37,1.4,0.5,1.61,1.4,0.39,0.957142857,0.972222222,0.291044776,1.15,0  ],
    [1.41,1.38,0.37,1.42,0.54,1.77,1.41,0.41,0.971830986,1,0.297101449,1.255319149,0  ],
    [1.44,1.43,0.36,1.49,0.49,1.72,1.49,0.47,0.959731544,1.034722222,0.328671329,1.154362416,0  ],
    [1.45,1.37,0.34,1.4,0.42,1.5,1.38,0.46,0.978571429,0.951724138,0.335766423,1.086956522,0  ],
    [1.46,1.4,0.36,1.43,0.59,1.71,1.44,0.44,0.979020979,0.98630137,0.314285714,1.1875,0  ],
    [1.4,1.32,0.26,1.37,0.47,1.68,1.36,0.45,0.96350365,0.971428571,0.340909091,1.235294118,0  ],
    [1.46,1.37,0.33,1.43,0.54,1.75,1.41,0.46,0.958041958,0.965753425,0.335766423,1.241134752,1  ],
    [1.43,1.34,0.35,1.42,0.54,1.58,1.41,0.48,0.943661972,0.986013986,0.358208955,1.120567376,1  ],
    [1.45,1.37,0.33,1.36,0.47,1.66,1.36,0.46,1.007352941,0.937931034,0.335766423,1.220588235,0  ],
    [1.45,1.37,0.39,1.43,0.52,1.74,1.41,0.47,0.958041958,0.972413793,0.343065693,1.234042553,1  ],
    [1.42,1.41,0.36,1.38,0.57,1.6,1.4,0.43,1.02173913,0.985915493,0.304964539,1.142857143,0  ],
    [1.4,1.34,0.31,1.36,0.47,1.6,1.36,0.45,0.985294118,0.971428571,0.335820896,1.176470588,0  ],
    [1.45,1.41,0.33,0.46,0.48,1.6,1.38,0.43,3.065217391,0.951724138,0.304964539,1.15942029,0  ],
    [1.48,1.44,0.35,1.49,0.56,1.69,1.46,0.42,0.966442953,0.986486486,0.291666667,1.157534247,0  ],
    [1.44,0.48,0.33,1.39,0.45,1.64,1.35,0.46,0.345323741,0.9375,0.958333333,1.214814815,1  ],
    [1.4,1.31,0.34,1.4,0.54,1.5,1.37,0.47,0.935714286,0.978571429,0.358778626,1.094890511,1  ],
    [1.42,1.36,0.33,1.45,0.55,1.77,1.42,0.46,0.937931034,1,0.338235294,1.246478873,0  ],
    [1.38,1.34,0.3,1.38,0.46,1.56,1.36,0.39,0.971014493,0.985507246,0.291044776,1.147058824,1  ],
    [1.5,1.4,0.38,1.41,0.48,1.68,1.4,0.5,0.992907801,0.933333333,0.357142857,1.2,0  ],
    [1.38,1.33,0.25,1.36,0.42,1.5,1.35,0.37,0.977941176,0.97826087,0.278195489,1.111111111,0  ],
    [1.46,1.4,0.37,1.44,0.56,1.76,1.43,0.43,0.972222222,0.979452055,0.307142857,1.230769231,0  ],
    [1.37,1.35,0.3,1.38,0.47,1.57,1.37,0.39,0.97826087,1,0.288888889,1.145985401,1  ],
    [1.43,1.35,0.31,1.42,0.47,1.62,1.39,0.48,0.950704225,0.972027972,0.355555556,1.165467626,0  ],
    [1.5,1.4,0.34,1.4,0.47,1.61,1.39,0.39,1,0.926666667,0.278571429,1.158273381,0  ],
    [1.4,1.3,0.31,1.38,0.57,1.55,1.36,0.45,0.942028986,0.971428571,0.346153846,1.139705882,1  ],
    [1.49,1.41,0.36,1.42,0.5,1.55,1.41,0.5,0.992957746,0.946308725,0.354609929,1.09929078,0  ],
    [1.43,1.32,0.33,1.38,0.57,1.48,1.36,0.44,0.956521739,0.951048951,0.333333333,1.088235294,1  ],
    [1.49,1.39,0.39,1.4,0.55,1.62,1.38,0.46,0.992857143,0.926174497,0.330935252,1.173913043,0  ],
    [1.48,1.32,0.33,1.41,0.59,1.59,1.41,0.49,0.936170213,0.952702703,0.371212121,1.127659574,1  ],
    [1.43,1.35,0.38,1.43,0.58,1.6,1.42,0.47,0.944055944,0.993006993,0.348148148,1.126760563,1  ],
    [1.43,1.35,0.32,1.42,0.59,1.56,1.4,0.46,0.950704225,0.979020979,0.340740741,1.114285714,1  ],
    [1.43,1.36,0.32,1.41,0.53,1.6,1.4,0.41,0.964539007,0.979020979,0.301470588,1.142857143,0  ],
    [1.38,1.36,0.32,1.41,0.44,1.64,1.37,0.4,0.964539007,0.992753623,0.294117647,1.197080292,0  ],
    [1.55,1.34,0.4,1.44,0.51,1.77,1.43,0.54,0.930555556,0.922580645,0.402985075,1.237762238,1  ],
    [1.38,1.39,0.34,1.42,0.46,1.69,1.38,0.45,0.978873239,1,0.323741007,1.224637681,0  ],
    [1.4,1.33,0.33,1.35,0.45,1.68,1.35,0.45,0.985185185,0.964285714,0.338345865,1.244444444,0  ],
    [1.4,1.33,0.37,1.38,0.55,1.68,1.38,0.43,0.963768116,0.985714286,0.323308271,1.217391304,1  ],
    [1.44,1.38,0.36,1.43,0.53,1.65,1.41,0.49,0.965034965,0.979166667,0.355072464,1.170212766,0  ],
    [1.41,1.36,0.3,1.42,0.58,1.77,1.4,0.46,0.957746479,0.992907801,0.338235294,1.264285714,1  ],
    [1.47,1.34,0.32,1.42,0.56,1.57,1.39,0.43,0.943661972,0.945578231,0.320895522,1.129496403,1  ],
    [1.46,1.4,0.37,1.45,0.55,1.73,1.43,0.41,0.965517241,0.979452055,0.292857143,1.20979021,0  ],
    [1.52,1.45,0.38,1.48,0.61,1.8,1.47,0.45,0.97972973,0.967105263,0.310344828,1.224489796,0  ],
    [1.43,1.37,0.36,1.42,0.54,1.7,1.41,0.42,0.964788732,0.986013986,0.306569343,1.205673759,1  ],
    [1.4,1.34,0.34,1.39,0.46,1.67,1.31,0.45,0.964028777,0.935714286,0.335820896,1.27480916,1  ],
    [1.43,1.39,0.26,1.42,0.45,1.69,1.4,0.39,0.978873239,0.979020979,0.28057554,1.207142857,0  ],
    [1.43,1.37,0.35,1.42,0.55,1.71,1.4,0.47,0.964788732,0.979020979,0.343065693,1.221428571,0  ],
    [1.43,1.37,0.31,1.41,0.54,1.52,1.4,0.4,0.971631206,0.979020979,0.291970803,1.085714286,0  ],
    [1.45,1.39,0.32,1.43,0.61,1.55,1.42,0.44,0.972027972,0.979310345,0.316546763,1.091549296,0  ],
    [1.38,1.3,0.31,1.39,0.51,1.52,1.36,0.43,0.935251799,0.985507246,0.330769231,1.117647059,1  ],
    [1.45,1.37,0.37,1.45,0.59,1.72,1.43,0.46,0.944827586,0.986206897,0.335766423,1.202797203,0  ],
    [1.48,1.43,0.35,1.46,0.52,1.77,1.46,0.46,0.979452055,0.986486486,0.321678322,1.212328767,1  ],
    [1.39,1.37,0.31,1.4,0.49,1.6,1.34,0.42,0.978571429,0.964028777,0.306569343,1.194029851,1  ],
    [1.39,1.36,0.32,1.39,0.53,1.58,1.38,0.4,0.978417266,0.992805755,0.294117647,1.144927536,1  ],
    [1.5,1.46,0.38,1.52,0.56,1.64,1.52,0.45,0.960526316,1.013333333,0.308219178,1.078947368,0  ],
    [1.43,1.37,0.31,1.41,0.58,1.65,1.4,0.39,0.971631206,0.979020979,0.284671533,1.178571429,1  ],
    [1.38,1.33,0.32,1.38,0.56,1.53,1.35,0.42,0.963768116,0.97826087,0.315789474,1.133333333,1  ],
    [1.45,1.39,0.38,1.43,0.58,1.68,1.43,0.45,0.972027972,0.986206897,0.323741007,1.174825175,0  ],
    [1.48,1.34,0.35,1.42,0.55,1.73,1.41,0.47,0.943661972,0.952702703,0.350746269,1.226950355,1  ],
    [1.45,1.34,0.4,1.41,0.57,1.67,1.41,0.47,0.95035461,0.972413793,0.350746269,1.184397163,0  ],
    [1.42,1.37,0.27,1.39,0.42,1.58,1.38,0.39,0.985611511,0.971830986,0.284671533,1.144927536,0  ],
    [1.39,1.29,0.35,1.36,0.55,1.59,1.35,0.49,0.948529412,0.971223022,0.379844961,1.177777778,1  ],
    [1.43,1.33,0.28,1.37,0.46,1.6,1.37,0.4,0.97080292,0.958041958,0.30075188,1.167883212,0  ],
    [1.4,1.43,0.33,1.43,0.47,1.73,1.39,0.43,1,0.992857143,0.300699301,1.244604317,0  ],
    [1.46,1.32,0.33,1.4,0.57,1.6,1.38,0.46,0.942857143,0.945205479,0.348484848,1.15942029,1  ],
    [1.37,1.33,0.28,1.34,0.43,1.59,1.35,0.4,0.992537313,0.98540146,0.30075188,1.177777778,0  ],
    [1.42,1.36,0.35,1.44,0.6,1.6,1.42,0.45,0.944444444,1,0.330882353,1.126760563,0  ],
    [1.4,1.31,0.31,1.28,0.4,1.51,1.31,0.45,1.0234375,0.935714286,0.34351145,1.152671756,1  ],
    [1.42,1.27,0.34,1.39,0.55,1.59,1.35,0.47,0.913669065,0.950704225,0.37007874,1.177777778,1  ],
    [1.35,1.35,0.34,1.37,0.45,1.5,1.37,0.46,0.98540146,1.014814815,0.340740741,1.094890511,0  ],
    [1.47,1.39,0.32,1.44,0.48,1.68,1.43,0.42,0.965277778,0.972789116,0.302158273,1.174825175,0  ],
    [1.38,1.33,0.27,1.37,0.44,1.56,1.37,0.38,0.97080292,0.992753623,0.285714286,1.138686131,0  ],
    [1.45,1.39,0.38,1.43,0.58,1.68,1.43,0.45,0.972027972,0.986206897,0.323741007,1.174825175,0  ],
    [1.39,1.36,0.3,1.39,0.62,1.7,1.37,0.42,0.978417266,0.985611511,0.308823529,1.240875912,1  ],
    [1.42,1.33,0.3,1.34,0.45,1.61,1.35,0.38,0.992537313,0.950704225,0.285714286,1.192592593,0  ],
    [1.38,1.35,0.31,1.38,0.44,1.62,1.36,0.41,0.97826087,0.985507246,0.303703704,1.191176471,0  ],
    [1.5,1.44,0.34,1.47,0.46,1.72,1.43,0.45,0.979591837,0.953333333,0.3125,1.202797203,0  ],
    [1.41,1.34,0.33,1.38,0.57,1.65,1.37,0.43,0.971014493,0.971631206,0.320895522,1.204379562,1  ],
    [1.43,1.35,0.33,1.39,0.51,1.64,1.39,0.41,0.971223022,0.972027972,0.303703704,1.179856115,1  ],
    [1.49,1.4,0.35,1.46,0.44,1.73,1.45,0.45,0.95890411,0.973154362,0.321428571,1.193103448,1  ],
    [1.47,1.4,0.38,1.4,0.49,1.66,1.39,0.48,1,0.945578231,0.342857143,1.194244604,0  ],
    [1.48,1.38,0.35,1.46,0.6,1.65,1.45,0.46,0.945205479,0.97972973,0.333333333,1.137931034,0  ],
    [1.4,1.34,0.33,1.35,0.5,1.57,1.3,0.45,0.992592593,0.928571429,0.335820896,1.207692308,1  ],
    [1.42,1.35,0.32,1.34,0.55,1.6,1.32,0.46,1.007462687,0.929577465,0.340740741,1.212121212,1  ],
    [1.4,1.3,0.33,1.31,0.43,1.73,1.31,0.46,0.992366412,0.935714286,0.353846154,1.320610687,1  ],
    [1.45,1.36,0.33,1.36,0.45,1.53,1.35,0.46,1,0.931034483,0.338235294,1.133333333,0  ],
    [1.4,1.33,0.32,1.37,0.54,1.5,1.36,0.44,0.97080292,0.971428571,0.330827068,1.102941176,1  ],
    [1.47,1.39,0.33,1.42,0.49,1.63,1.33,0.43,0.978873239,0.904761905,0.309352518,1.22556391,1  ],
    [1.39,1.33,0.33,1.38,0.57,1.7,1.35,0.43,0.963768116,0.971223022,0.323308271,1.259259259,1  ],
    [1.45,1.4,0.36,1.45,0.64,1.88,1.43,0.43,0.965517241,0.986206897,0.307142857,1.314685315,0  ],
    [1.41,1.37,0.32,1.42,0.46,1.68,1.39,0.42,0.964788732,0.985815603,0.306569343,1.208633094,0  ],
    [1.42,1.36,0.34,1.37,0.48,1.6,1.36,0.42,0.99270073,0.957746479,0.308823529,1.176470588,0  ],
    [1.4,1.37,0.41,1.39,0.56,1.58,1.37,0.48,0.985611511,0.978571429,0.350364964,1.153284672,1  ],
    [1.42,1.36,0.34,1.39,0.46,1.65,1.34,0.46,0.978417266,0.943661972,0.338235294,1.231343284,1  ],
    [1.4,1.3,0.33,1.39,0.57,1.47,1.37,0.44,0.935251799,0.978571429,0.338461538,1.072992701,1  ],
    [1.4,1.36,0.33,1.4,0.55,1.75,1.37,0.42,0.971428571,0.978571429,0.308823529,1.277372263,1  ],
    [1.43,1.33,0.37,1.42,0.57,1.72,1.41,0.46,0.936619718,0.986013986,0.345864662,1.219858156,1  ],
    [1.46,1.38,0.35,1.44,0.59,1.6,1.43,0.52,0.958333333,0.979452055,0.376811594,1.118881119,1  ],
    [1.45,1.42,0.36,1.44,0.47,1.64,1.32,0.46,0.986111111,0.910344828,0.323943662,1.242424242,1  ],
    [1.42,1.36,0.34,1.43,0.59,1.57,1.42,0.46,0.951048951,1,0.338235294,1.105633803,1  ],
    [1.45,1.36,0.36,1.48,0.59,1.61,1.46,0.36,0.918918919,1.006896552,0.264705882,1.102739726,0  ],
    [1.52,1.45,0.38,1.48,0.61,1.8,1.47,0.45,0.97972973,0.967105263,0.310344828,1.224489796,0  ],
    [1.43,1.32,0.3,1.37,0.45,1.66,1.36,0.44,0.96350365,0.951048951,0.333333333,1.220588235,0  ],
    [1.48,1.41,0.36,1.49,0.56,1.7,1.49,0.45,0.946308725,1.006756757,0.319148936,1.140939597,0  ],
    [1.42,1.34,0.32,1.37,0.57,1.52,1.36,0.41,0.97810219,0.957746479,0.305970149,1.117647059,1  ],
    [1.41,1.35,0.35,1.4,0.55,1.6,1.41,0.44,0.964285714,1,0.325925926,1.134751773,0  ],
    [1.35,1.32,0.31,1.33,0.51,1.57,1.31,0.46,0.992481203,0.97037037,0.348484848,1.198473282,1  ],
    [1.45,1.38,0.33,1.41,0.43,1.57,1.34,0.42,0.978723404,0.924137931,0.304347826,1.171641791,1  ],
    [1.39,1.38,0.33,1.45,0.54,1.68,1.44,0.44,0.951724138,1.035971223,0.31884058,1.166666667,0  ],
    [1.45,1.3,0.26,1.35,0.46,1.64,1.35,0.42,0.962962963,0.931034483,0.323076923,1.214814815,0  ],
    [1.5,1.34,0.32,1.41,0.59,1.6,1.39,0.45,0.95035461,0.926666667,0.335820896,1.151079137,1  ],
    [1.37,1.34,0.3,1.39,0.48,1.59,1.37,0.4,0.964028777,1,0.298507463,1.160583942,1  ],
    [1.52,1.48,0.36,1.52,0.57,1.8,1.5,0.43,0.973684211,0.986842105,0.290540541,1.2,0  ],
    [1.35,1.3,0.3,1.37,0.45,1.62,1.35,0.42,0.948905109,1,0.323076923,1.2,0  ],
    [1.47,1.38,0.43,1.44,0.59,1.59,1.44,0.54,0.958333333,0.979591837,0.391304348,1.104166667,0  ],
    [1.47,1.41,0.3,1.43,0.54,1.6,1.45,0.42,0.986013986,0.986394558,0.29787234,1.103448276,1  ],
    [1.49,1.38,0.39,1.41,0.61,1.56,1.4,0.46,0.978723404,0.939597315,0.333333333,1.114285714,0  ],
    [1.4,1.36,0.33,1.39,0.44,1.66,1.38,0.44,0.978417266,0.985714286,0.323529412,1.202898551,0  ],
    [1.4,1.33,0.28,1.37,0.47,1.7,1.37,0.47,0.97080292,0.978571429,0.353383459,1.240875912,0  ],
    [1.38,1.33,0.26,1.37,0.43,1.61,1.35,0.42,0.97080292,0.97826087,0.315789474,1.192592593,0  ],
    [1.44,1.38,0.36,1.43,0.48,1.68,1.43,0.47,0.965034965,0.993055556,0.34057971,1.174825175,0  ],
    [1.4,1.35,0.27,1.37,0.43,1.61,1.38,0.38,0.98540146,0.985714286,0.281481481,1.166666667,0  ],
    [1.35,1.33,0.31,1.38,0.44,1.58,1.36,0.41,0.963768116,1.007407407,0.308270677,1.161764706,0  ],
    [1.48,1.36,0.27,1.38,0.54,1.71,1.4,0.46,0.985507246,0.945945946,0.338235294,1.221428571,1  ],
    [1.35,1.29,0.3,1.35,0.43,1.6,1.35,0.39,0.955555556,1,0.302325581,1.185185185,0  ],
    [1.44,1.38,0.35,1.37,0.58,1.67,1.36,0.47,1.00729927,0.944444444,0.34057971,1.227941176,0  ],
    [1.48,1.41,0.37,1.47,0.57,1.69,1.46,0.45,0.959183673,0.986486486,0.319148936,1.157534247,1  ],
    [1.45,1.38,0.36,1.42,0.56,1.6,1.41,0.45,0.971830986,0.972413793,0.326086957,1.134751773,0  ],
    [1.41,1.38,0.33,1.4,0.57,1.6,1.41,0.41,0.985714286,1,0.297101449,1.134751773,0  ],
    [1.44,1.36,0.32,1.42,0.61,1.66,1.41,0.44,0.957746479,0.979166667,0.323529412,1.177304965,1  ],
    [1.48,1.34,0.37,1.4,0.58,1.6,1.4,0.47,0.957142857,0.945945946,0.350746269,1.142857143,1  ],
    [1.44,1.38,0.3,1.43,0.47,1.64,1.34,0.42,0.965034965,0.930555556,0.304347826,1.223880597,1  ],
    [1.48,1.32,0.32,1.4,0.57,1.66,1.37,0.46,0.942857143,0.925675676,0.348484848,1.211678832,1  ],
    [1.45,1.33,0.38,1.42,0.56,1.68,1.4,0.46,0.936619718,0.965517241,0.345864662,1.2,0  ],
    [1.43,1.35,0.3,1.38,0.46,1.67,1.38,0.44,0.97826087,0.965034965,0.325925926,1.210144928,0  ],
    [1.46,1.34,0.37,1.44,0.57,1.61,1.41,0.46,0.930555556,0.965753425,0.343283582,1.141843972,0  ],
    [1.38,1.36,0.27,1.4,0.48,1.61,1.4,0.37,0.971428571,1.014492754,0.272058824,1.15,0  ],
    [1.38,1.33,0.33,1.37,0.55,1.52,1.35,0.41,0.97080292,0.97826087,0.308270677,1.125925926,0  ],
    [1.32,1.36,0.24,1.37,0.45,1.63,1.37,0.38,0.99270073,1.037878788,0.279411765,1.189781022,0  ],
    [1.4,1.31,0.21,1.35,0.4,1.6,1.35,0.35,0.97037037,0.964285714,0.267175573,1.185185185,0  ]
  ]

  module.exports = treinox