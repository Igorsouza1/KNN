const treinox = [
    [1.33,1.31,0.32,1.33,0.46,1.6,1.3,0.45,0.98,0.98,1,1.230769231,1  ],
    [1.43,1.37,0.27,1.42,0.51,1.66,1.41,0.42,1.3,0.99,0.81,1.177304965,1  ],
    [1.53,1.36,0.42,1.44,0.58,1.76,1.44,0.51,1.5,0.941176471,0.894736842,1.222222222,0  ],
    [1.34,1.36,0.34,1.37,0.43,1.57,1.36,0.44,1.05,1.01,0.96,1.154411765,0  ],
    [1.37,1.33,0.3,1.37,0.46,1.54,1.35,0.41,1.22,0.99,0.84,1.140740741,1  ],
    [1.42,1.36,0.27,1.4,0.45,1.55,1.38,0.4,1.24,0.97,0.77,1.123188406,0  ],
    [1.46,1.36,0.35,1.47,0.58,1.66,1.48,0.46,1.33,1.01,0.77,1.121621622,0  ],
    [1.45,1.4,0.34,1.45,0.56,1.66,1.44,0.47,1.18,0.99,0.81,1.152777778,0  ],
    [1.41,1.38,0.38,1.44,0.54,1.59,1.41,0.49,1.25,1,0.98,1.127659574,0  ],
    [1.36,0.51,0.33,1.4,0.47,1.63,1.39,0.45,1.06,1.02,0.88,1.172661871,1  ],
    [1.42,1.34,0.27,1.36,0.45,1.65,1.37,0.4,1.19,0.96,0.8,1.204379562,0  ],
    [1.5,1.42,0.36,1.5,0.55,1.69,1.49,0.45,1.22,0.99,0.75,1.134228188,0  ],
    [1.4,1.33,0.3,1.36,0.43,1.62,1.35,0.41,1.29,0.96,0.76,1.2,0  ],
    [1.43,1.36,0.34,1.4,0.46,1.74,1.42,0.46,1.18,0.99,0.88,1.225352113,0  ],
    [1.46,1.4,0.36,1.43,0.47,1.67,1.3,0.42,1.05,0.89,0.91,1.284615385,1  ],
    [1.39,1.34,0.28,1.37,0.43,1.5,1.36,0.42,1.04,0.98,0.89,1.102941176,0  ],
    [1.43,1.36,0.38,1.41,0.55,1.68,1.41,0.44,1.36,0.99,0.73,1.191489362,1  ],
    [1.43,1.35,0.37,1.4,0.53,1.69,1.41,0.44,1.34,0.99,0.75,1.19858156,1  ],
    [1.4,1.32,0.34,1.36,0.45,1.62,1.35,0.45,1.2,0.96,0.83,1.2,0  ],
    [1.42,1.32,0.35,1.39,0.55,1.63,1.37,0.44,1.24,0.96,0.77,1.189781022,1  ],
    [1.37,1.35,0.3,1.39,0.46,1.63,1.37,0.42,1.17,1,0.86,1.189781022,0  ],
    [1.43,1.32,0.37,1.4,0.59,1.69,1.38,0.48,1.19,0.97,0.84,1.224637681,1  ],
    [1.47,1.34,0.38,1.42,0.6,1.69,1.41,0.46,1.2,0.96,0.75,1.19858156,0  ],
    [1.48,1.38,0.38,1.47,0.61,1.61,1.45,0.51,1.33,0.98,0.74,1.110344828,1  ],
    [1.45,1.4,0.34,1.45,0.53,1.62,1.43,0.4,1.38,0.99,0.74,1.132867133,0  ],
    [1.38,1.34,0.27,1.37,0.44,1.77,1.37,0.42,1.14,0.99,0.84,1.291970803,0  ],
    [1.4,1.41,0.34,1.39,0.47,1.62,1.39,0.4,0.98,0.99,1,1.165467626,0  ],
    [1.45,1.38,0.34,1.43,0.46,1.61,1.33,0.44,1.25,0.92,0.8,1.210526316,1  ],
    [1.44,1.41,0.34,1.44,0.58,1.61,1.43,0.43,1.24,0.99,0.83,1.125874126,0  ],
    [1.47,1.35,0.36,1.42,0.55,1.66,1.43,0.49,1.46,0.97,0.82,1.160839161,1  ],
    [1.46,1.29,0.35,1.39,0.5,1.64,1.36,0.46,1.51,0.93,0.82,1.205882353,1  ],
    [1.43,1.37,0.36,1.43,0.54,1.68,1.41,0.47,1.21,0.99,0.81,1.191489362,1  ],
    [1.45,1.38,0.31,1.39,0.46,1.63,1.37,0.41,1.11,0.94,0.82,1.189781022,0  ],
    [1.4,1.32,0.34,1.36,0.45,1.62,1.35,0.45,1.2,0.96,0.83,1.2,0  ],
    [1.48,1.38,0.34,1.44,0.51,1.57,1.42,0.46,1.4,0.96,0.66,1.105633803,1  ],
    [1.42,1.34,0.37,1.4,0.56,1.58,1.39,0.44,1.37,0.98,0.75,1.136690647,1  ],
    [1.45,1.4,0.34,1.44,0.53,1.7,1.44,0.42,1.28,0.99,0.76,1.180555556,0  ],
    [1.41,1.39,0.32,1.42,0.54,1.64,1.4,0.41,1.32,0.99,0.77,1.171428571,1  ],
    [1.39,1.35,0.34,1.38,0.54,1.62,1.37,0.42,1.32,0.99,0.78,1.182481752,1  ],
    [1.4,1.35,0.3,1.39,0.45,1.66,1.31,0.43,1.23,0.94,0.8,1.267175573,1  ],
    [1.39,1.31,0.25,1.37,0.46,1.58,1.36,0.4,0.96,0.98,0.8,1.161764706,0  ],
    [1.4,1.34,0.32,1.4,0.56,1.65,1.37,0.41,1.27,0.98,0.79,1.204379562,1  ],
    [1.47,1.43,0.32,1.47,0.56,1.6,1.45,0.41,1.3,0.99,0.79,1.103448276,0  ],
    [1.47,1.44,0.36,1.47,0.45,1.61,1.33,0.47,1.02,0.9,1.02,1.210526316,1  ],
    [1.42,1.38,0.33,1.41,0.53,1.73,1.39,0.4,1.41,0.98,0.69,1.244604317,1  ],
    [1.31,1.33,0.28,1.36,0.45,1.61,1.35,0.4,1.16,1.03,0.8,1.192592593,0  ],
    [1.45,1.42,0.33,1.45,0.55,1.86,1.43,0.42,1.26,0.99,0.79,1.300699301,1  ],
    [1.42,1.35,0.3,1.39,0.44,1.6,1.37,0.43,1.07,0.96,0.91,1.167883212,0  ],
    [1.36,1.35,0.28,1.38,0.45,1.6,1.36,0.44,1.19,1,0.88,1.176470588,0  ],
    [1.42,1.38,0.36,1.44,0.54,1.7,1.42,0.44,1.23,1,0.83,1.197183099,0  ],
    [1.41,1.37,0.32,1.4,0.45,1.64,1.39,0.46,1.27,0.99,0.82,1.179856115,0  ],
    [1.44,1.36,0.38,1.42,0.55,1.69,1.42,0.44,1.34,0.99,0.75,1.190140845,1  ],
    [1.41,1.33,0.3,1.35,0.44,1.69,1.36,0.4,1.21,0.96,0.77,1.242647059,0  ],
    [1.39,1.31,0.26,1.35,0.46,1.6,1.35,0.42,1.05,0.97,0.91,1.185185185,0  ],
    [1.37,1.37,0.27,1.38,0.45,1.5,1.37,0.38,1.12,1,0.79,1.094890511,0  ],
    [1.51,1.36,0.38,1.45,0.62,1.64,1.43,0.46,1.34,0.95,0.69,1.146853147,1  ],
    [1.44,1.35,0.31,1.41,0.55,1.65,1.4,0.45,1.24,0.97,0.8,1.178571429,1  ],
    [1.41,1.41,0.3,1.45,0.49,1.68,1.45,0.42,1.24,1.03,0.74,1.15862069,0  ],
    [1.39,1.31,0.31,1.39,0.54,1.46,1.36,0.42,1.4,0.98,0.67,1.073529412,1  ],
    [1.33,1.31,0.32,1.33,0.46,1.6,1.3,0.45,0.98,0.98,1,1.230769231,0  ],
    [1.41,1.36,0.32,1.4,0.54,1.59,1.38,0.42,1.4,0.98,0.75,1.152173913,1  ],
    [1.43,1.37,0.31,1.42,0.52,1.72,1.41,0.45,1.38,0.99,0.82,1.219858156,1  ],
    [1.48,1.41,0.36,1.42,0.58,1.63,1.42,0.45,1.42,0.96,0.74,1.147887324,0  ],
    [1.48,1.31,0.36,1.39,0.58,1.59,1.37,0.45,1.31,0.93,0.7,1.160583942,1  ],
    [1.43,1.4,0.32,1.44,0.56,1.65,1.41,0.43,1.25,0.99,0.78,1.170212766,1  ],
    [1.35,1.33,0.33,1.34,0.44,1.48,1.33,0.42,1.12,0.99,0.88,1.112781955,0  ],
    [1.44,1.34,0.38,1.43,0.69,1.6,1.4,0.48,1.41,0.97,0.74,1.142857143,0  ],
    [1.4,1.31,0.33,1.41,0.56,1.58,1.37,0.46,1.36,0.98,0.72,1.153284672,1  ],
    [1.43,1.35,0.32,1.41,0.54,1.58,1.42,0.42,1.21,0.99,0.82,1.112676056,0  ],
    [1.45,1.38,0.39,1.45,0.48,1.61,1.43,0.48,1.48,0.99,0.71,1.125874126,0  ],
    [1.41,1.38,0.28,1.39,0.44,1.6,1.38,0.4,1.13,0.98,0.78,1.15942029,0  ],
    [1.44,1.42,0.37,1.44,0.49,1.64,1.32,0.46,0.85,0.92,1.12,1.242424242,1  ],
    [1.41,1.36,0.36,1.4,0.56,1.64,1.38,0.4,1.35,0.98,0.74,1.188405797,1  ],
    [1.45,1.4,0.35,1.45,0.57,1.67,1.43,0.41,1.31,0.99,0.75,1.167832168,0  ],
    [1.38,1.33,0.26,1.37,0.43,1.61,1.35,0.42,1.21,0.98,0.81,1.192592593,0  ],
    [1.48,1.42,0.36,1.49,0.49,1.69,1.49,0.47,1.09,1.01,0.98,1.134228188,0  ],
    [1.38,1.35,0.29,1.38,0.45,1.66,1.36,0.4,1.28,0.99,0.73,1.220588235,0  ],
    [1.33,1.29,0.32,1.35,0.48,1.51,1.31,0.42,1.11,0.98,0.84,1.152671756,0  ],
    [1.41,1.41,0.3,1.45,0.49,1.68,1.45,0.42,1.24,1.03,0.74,1.15862069,0  ],
    [1.43,1.31,0.3,1.35,0.45,1.62,1.35,0.41,1,0.94,0.91,1.2,0  ],
    [1.42,1.36,0.29,1.39,0.48,1.69,1.4,0.4,1.2,0.99,0.75,1.207142857,0  ],
    [1.3,1.32,0.29,1.35,0.45,1.66,1.35,0.42,1.28,1.04,0.76,1.22962963,0  ],
    [1.44,1.34,0.37,1.4,0.5,1.61,1.4,0.39,1.31,0.97,0.71,1.15,0  ],
    [1.41,1.38,0.37,1.42,0.54,1.77,1.41,0.41,1.25,1,0.82,1.255319149,0  ],
    [1.44,1.43,0.36,1.49,0.49,1.72,1.49,0.47,1.09,1.03,0.96,1.154362416,0  ],
    [1.45,1.37,0.34,1.4,0.42,1.5,1.38,0.46,1.24,0.95,0.88,1.086956522,0  ],
    [1.46,1.4,0.36,1.43,0.59,1.71,1.44,0.44,1.19,0.99,0.88,1.1875,0  ],
    [1.4,1.32,0.26,1.37,0.47,1.68,1.36,0.45,0.96,0.97,1,1.235294118,0  ],
    [1.46,1.37,0.33,1.43,0.54,1.75,1.41,0.46,1.26,0.97,0.87,1.241134752,1  ],
    [1.43,1.34,0.35,1.42,0.54,1.58,1.41,0.48,1.36,0.99,0.79,1.120567376,1  ],
    [1.45,1.37,0.33,1.36,0.47,1.66,1.36,0.46,1.1,0.94,1,1.220588235,0  ],
    [1.45,1.37,0.39,1.43,0.52,1.74,1.41,0.47,1.22,0.97,0.84,1.234042553,1  ],
    [1.42,1.41,0.36,1.38,0.57,1.6,1.4,0.43,1.4,0.99,0.68,1.142857143,0  ],
    [1.4,1.34,0.31,1.36,0.47,1.6,1.36,0.45,1,0.97,0.96,1.176470588,0  ],
    [1.45,1.41,0.33,0.46,0.48,1.6,1.38,0.43,1.28,0.95,0.73,1.15942029,0  ],
    [1.48,1.44,0.35,1.49,0.56,1.69,1.46,0.42,1.33,0.99,0.74,1.157534247,0  ],
    [1.44,0.48,0.33,1.39,0.45,1.64,1.35,0.46,1.07,0.94,0.96,1.214814815,1  ],
    [1.4,1.31,0.34,1.4,0.54,1.5,1.37,0.47,1.47,0.98,0.71,1.094890511,1  ],
    [1.42,1.36,0.33,1.45,0.55,1.77,1.42,0.46,1.18,1,0.88,1.246478873,0  ],
    [1.38,1.34,0.3,1.38,0.46,1.56,1.36,0.39,1.26,0.99,0.81,1.147058824,1  ],
    [1.5,1.4,0.38,1.41,0.48,1.68,1.4,0.5,1.15,0.93,0.93,1.2,0  ],
    [1.38,1.33,0.25,1.36,0.42,1.5,1.35,0.37,1.14,0.98,0.77,1.111111111,0  ],
    [1.46,1.4,0.37,1.44,0.56,1.76,1.43,0.43,1.21,0.98,0.83,1.230769231,0  ],
    [1.37,1.35,0.3,1.38,0.47,1.57,1.37,0.39,1.2,1,0.81,1.145985401,1  ],
    [1.43,1.35,0.31,1.42,0.47,1.62,1.39,0.48,1.02,0.97,0.98,1.165467626,0  ],
    [1.5,1.4,0.34,1.4,0.47,1.61,1.39,0.39,1.4,0.93,0.65,1.158273381,0  ],
    [1.4,1.3,0.31,1.38,0.57,1.55,1.36,0.45,1.35,0.97,0.68,1.139705882,1  ],
    [1.49,1.41,0.36,1.42,0.5,1.55,1.41,0.5,1.06,0.95,0.91,1.09929078,0  ],
    [1.43,1.32,0.33,1.38,0.57,1.48,1.36,0.44,1.52,0.95,0.66,1.088235294,1  ],
    [1.49,1.39,0.39,1.4,0.55,1.62,1.38,0.46,1.23,0.93,0.78,1.173913043,0  ],
    [1.48,1.32,0.33,1.41,0.59,1.59,1.41,0.49,1.43,0.95,0.7,1.127659574,1  ],
    [1.43,1.35,0.38,1.43,0.58,1.6,1.42,0.47,1.46,0.99,0.67,1.126760563,1  ],
    [1.43,1.35,0.32,1.42,0.59,1.56,1.4,0.46,1.29,0.98,0.7,1.114285714,1  ],
    [1.43,1.36,0.32,1.41,0.53,1.6,1.4,0.41,1.21,0.98,0.79,1.142857143,0  ],
    [1.38,1.36,0.32,1.41,0.44,1.64,1.37,0.4,1.11,0.99,0.82,1.197080292,0  ],
    [1.55,1.34,0.4,1.44,0.51,1.77,1.43,0.54,1.1,0.922580645,0.9,1.237762238,1  ],
    [1.38,1.39,0.34,1.42,0.46,1.69,1.38,0.45,1.02,1,1.05,1.224637681,0  ],
    [1.4,1.33,0.33,1.35,0.45,1.68,1.35,0.45,0.84,0.96,1.07,1.244444444,0  ],
    [1.4,1.33,0.37,1.38,0.55,1.68,1.38,0.43,1.37,0.99,0.73,1.217391304,1  ],
    [1.44,1.38,0.36,1.43,0.53,1.65,1.41,0.49,1.29,0.98,0.91,1.170212766,0  ],
    [1.41,1.36,0.3,1.42,0.58,1.77,1.4,0.46,1.28,0.99,0.84,1.264285714,1  ],
    [1.47,1.34,0.32,1.42,0.56,1.57,1.39,0.43,1.13,0.95,0.8,1.129496403,1  ],
    [1.46,1.4,0.37,1.45,0.55,1.73,1.43,0.41,1.24,0.98,0.8,1.20979021,0  ],
    [1.52,1.45,0.38,1.48,0.61,1.8,1.47,0.45,1.22,0.97,0.8,1.224489796,0  ],
    [1.43,1.37,0.36,1.42,0.54,1.7,1.41,0.42,1.26,0.99,0.79,1.205673759,1  ],
    [1.4,1.34,0.34,1.39,0.46,1.67,1.31,0.45,1.12,0.94,0.98,1.27480916,1  ],
    [1.43,1.39,0.26,1.42,0.45,1.69,1.4,0.39,1.24,0.98,0.75,1.207142857,0  ],
    [1.43,1.37,0.35,1.42,0.55,1.71,1.4,0.47,1.26,0.98,0.81,1.221428571,0  ],
    [1.43,1.37,0.31,1.41,0.54,1.52,1.4,0.4,1.32,0.98,0.75,1.085714286,0  ],
    [1.45,1.39,0.32,1.43,0.61,1.55,1.42,0.44,1.23,0.98,0.81,1.091549296,0  ],
    [1.38,1.3,0.31,1.39,0.51,1.52,1.36,0.43,1.25,0.99,0.78,1.117647059,1  ],
    [1.45,1.37,0.37,1.45,0.59,1.72,1.43,0.46,1.26,0.99,0.78,1.202797203,0  ],
    [1.48,1.43,0.35,1.46,0.52,1.77,1.46,0.46,1.27,0.99,0.74,1.212328767,1  ],
    [1.39,1.37,0.31,1.4,0.49,1.6,1.34,0.42,0.98,0.96,0.95,1.194029851,1  ],
    [1.39,1.36,0.32,1.39,0.53,1.58,1.38,0.4,1.22,0.99,0.8,1.144927536,1  ],
    [1.5,1.46,0.38,1.52,0.56,1.64,1.52,0.45,1.44,1.01,0.69,1.078947368,0  ],
    [1.43,1.37,0.31,1.41,0.58,1.65,1.4,0.39,1.39,0.98,0.74,1.178571429,1  ],
    [1.38,1.33,0.32,1.38,0.56,1.53,1.35,0.42,1.21,0.98,0.81,1.133333333,1  ],
    [1.45,1.39,0.38,1.43,0.58,1.68,1.43,0.45,1.3,0.99,0.75,1.174825175,0  ],
    [1.48,1.34,0.35,1.42,0.55,1.73,1.41,0.47,1.38,0.95,0.85,1.226950355,1  ],
    [1.45,1.34,0.4,1.41,0.57,1.67,1.41,0.47,1.16,0.97,0.82,1.184397163,0  ],
    [1.42,1.37,0.27,1.39,0.42,1.58,1.38,0.39,1.24,0.97,0.75,1.144927536,0  ],
    [1.39,1.29,0.35,1.36,0.55,1.59,1.35,0.49,1.38,0.97,0.75,1.177777778,1  ],
    [1.43,1.33,0.28,1.37,0.46,1.6,1.37,0.4,1.02,0.96,0.91,1.167883212,0  ],
    [1.4,1.43,0.33,1.43,0.47,1.73,1.39,0.43,1.19,0.99,0.75,1.244604317,0  ],
    [1.46,1.32,0.33,1.4,0.57,1.6,1.38,0.46,1.23,0.95,0.87,1.15942029,1  ],
    [1.37,1.33,0.28,1.34,0.43,1.59,1.35,0.4,1.17,0.99,0.82,1.177777778,0  ],
    [1.42,1.36,0.35,1.44,0.6,1.6,1.42,0.45,1.45,1,0.7,1.126760563,0  ],
    [1.4,1.31,0.31,1.28,0.4,1.51,1.31,0.45,1.22,0.94,0.9,1.152671756,1  ],
    [1.42,1.27,0.34,1.39,0.55,1.59,1.35,0.47,1.41,0.95,0.81,1.177777778,1  ],
    [1.35,1.35,0.34,1.37,0.45,1.5,1.37,0.46,1.13,1.01,0.88,1.094890511,0  ],
    [1.47,1.39,0.32,1.44,0.48,1.68,1.43,0.42,1.22,0.97,0.75,1.174825175,0  ],
    [1.38,1.33,0.27,1.37,0.44,1.56,1.37,0.38,1.07,0.99,0.83,1.138686131,0  ],
    [1.45,1.39,0.38,1.43,0.58,1.68,1.43,0.45,1.3,0.99,0.75,1.174825175,0  ],
    [1.39,1.36,0.3,1.39,0.62,1.7,1.37,0.42,1.17,0.99,0.86,1.240875912,1  ],
    [1.42,1.33,0.3,1.34,0.45,1.61,1.35,0.38,1.16,0.95,0.76,1.192592593,0  ],
    [1.38,1.35,0.31,1.38,0.44,1.62,1.36,0.41,1.27,0.99,0.8,1.191176471,0  ],
    [1.5,1.44,0.34,1.47,0.46,1.72,1.43,0.45,1.28,0.95,0.76,1.202797203,0  ],
    [1.41,1.34,0.33,1.38,0.57,1.65,1.37,0.43,1.2,0.97,0.8,1.204379562,1  ],
    [1.43,1.35,0.33,1.39,0.51,1.64,1.39,0.41,1.3,0.97,0.67,1.179856115,1  ],
    [1.49,1.4,0.35,1.46,0.44,1.73,1.45,0.45,1.22,0.97,0.75,1.193103448,1  ],
    [1.47,1.4,0.38,1.4,0.49,1.66,1.39,0.48,0.98,0.95,1.02,1.194244604,0  ],
    [1.48,1.38,0.35,1.46,0.6,1.65,1.45,0.46,1.32,0.98,0.7,1.137931034,0  ],
    [1.4,1.34,0.33,1.35,0.5,1.57,1.3,0.45,1.11,0.93,0.88,1.207692308,1  ],
    [1.42,1.35,0.32,1.34,0.55,1.6,1.32,0.46,1.24,0.93,0.75,1.212121212,1  ],
    [1.4,1.3,0.33,1.31,0.43,1.73,1.31,0.46,1.07,0.94,1,1.320610687,1  ],
    [1.45,1.36,0.33,1.36,0.45,1.53,1.35,0.46,1.13,0.93,0.9,1.133333333,0  ],
    [1.4,1.33,0.32,1.37,0.54,1.5,1.36,0.44,1.34,0.97,0.8,1.102941176,1  ],
    [1.47,1.39,0.33,1.42,0.49,1.63,1.33,0.43,1.18,0.9,0.81,1.22556391,1  ],
    [1.39,1.33,0.33,1.38,0.57,1.7,1.35,0.43,1.29,0.97,0.8,1.259259259,1  ],
    [1.45,1.4,0.36,1.45,0.64,1.88,1.43,0.43,1.31,0.99,0.78,1.314685315,0  ],
    [1.41,1.37,0.32,1.42,0.46,1.68,1.39,0.42,1.16,0.99,0.84,1.208633094,0  ],
    [1.42,1.36,0.34,1.37,0.48,1.6,1.36,0.42,1.18,0.96,0.79,1.176470588,0  ],
    [1.4,1.37,0.41,1.39,0.56,1.58,1.37,0.48,1.26,0.98,0.76,1.153284672,1  ],
    [1.42,1.36,0.34,1.39,0.46,1.65,1.34,0.46,1.11,0.94,0.92,1.231343284,1  ],
    [1.4,1.3,0.33,1.39,0.57,1.47,1.37,0.44,1.39,0.98,0.69,1.072992701,1  ],
    [1.4,1.36,0.33,1.4,0.55,1.75,1.37,0.42,1.24,0.98,0.81,1.277372263,1  ],
    [1.43,1.33,0.37,1.42,0.57,1.72,1.41,0.46,1.19,0.99,0.82,1.219858156,1  ],
    [1.46,1.38,0.35,1.44,0.59,1.6,1.43,0.52,1.38,0.98,0.72,1.118881119,1  ],
    [1.45,1.42,0.36,1.44,0.47,1.64,1.32,0.46,1.02,0.91,1,1.242424242,1  ],
    [1.42,1.36,0.34,1.43,0.59,1.57,1.42,0.46,1.43,1,0.7,1.105633803,1  ],
    [1.45,1.36,0.36,1.48,0.59,1.61,1.46,0.36,1.26,1.01,0.62,1.102739726,0  ],
    [1.52,1.45,0.38,1.48,0.61,1.8,1.47,0.45,1.22,0.97,0.8,1.224489796,0  ],
    [1.43,1.32,0.3,1.37,0.45,1.66,1.36,0.44,0.9,0.95,1.02,1.220588235,0  ],
    [1.48,1.41,0.36,1.49,0.56,1.7,1.49,0.45,1.26,1.01,0.76,1.140939597,0  ],
    [1.42,1.34,0.32,1.37,0.57,1.52,1.36,0.41,1.33,0.96,0.73,1.117647059,1  ],
    [1.41,1.35,0.35,1.4,0.55,1.6,1.41,0.44,1.28,1,0.88,1.134751773,0  ],
    [1.35,1.32,0.31,1.33,0.51,1.57,1.31,0.46,1.11,0.97,0.92,1.198473282,1  ],
    [1.45,1.38,0.33,1.41,0.43,1.57,1.34,0.42,0.98,0.92,0.95,1.171641791,1  ],
    [1.39,1.38,0.33,1.45,0.54,1.68,1.44,0.44,1.13,1.04,0.83,1.166666667,0  ],
    [1.45,1.3,0.26,1.35,0.46,1.64,1.35,0.42,1,0.93,0.91,1.214814815,0  ],
    [1.5,1.34,0.32,1.41,0.59,1.6,1.39,0.45,1.05,0.93,0.75,1.151079137,1  ],
    [1.37,1.34,0.3,1.39,0.48,1.59,1.37,0.4,1.27,1,0.77,1.160583942,1  ],
    [1.52,1.48,0.36,1.52,0.57,1.8,1.5,0.43,1.21,0.99,0.84,1.2,0  ],
    [1.35,1.3,0.3,1.37,0.45,1.62,1.35,0.42,1.02,1,0.88,1.2,0  ],
    [1.47,1.38,0.43,1.44,0.59,1.59,1.44,0.54,1.24,0.98,0.79,1.104166667,0  ],
    [1.47,1.41,0.3,1.43,0.54,1.6,1.45,0.42,1.23,0.99,0.79,1.103448276,1  ],
    [1.49,1.38,0.39,1.41,0.61,1.56,1.4,0.46,1.66,0.94,0.63,1.114285714,0  ],
    [1.4,1.36,0.33,1.39,0.44,1.66,1.38,0.44,1.2,0.99,0.83,1.202898551,0  ],
    [1.4,1.33,0.28,1.37,0.47,1.7,1.37,0.47,0.94,0.98,1.07,1.240875912,0  ],
    [1.38,1.33,0.26,1.37,0.43,1.61,1.35,0.42,1.21,0.98,0.81,1.192592593,0  ],
    [1.44,1.38,0.36,1.43,0.48,1.68,1.43,0.47,1.13,0.99,0.87,1.174825175,0  ],
    [1.4,1.35,0.27,1.37,0.43,1.61,1.38,0.38,1.18,0.99,0.72,1.166666667,0  ],
    [1.35,1.33,0.31,1.38,0.44,1.58,1.36,0.41,1.19,1.01,0.82,1.161764706,0  ],
    [1.48,1.36,0.27,1.38,0.54,1.71,1.4,0.46,1.26,0.95,0.87,1.221428571,1  ],
    [1.35,1.29,0.3,1.35,0.43,1.6,1.35,0.39,1.27,1,0.75,1.185185185,0  ],
    [1.44,1.38,0.35,1.37,0.58,1.67,1.36,0.47,1.21,0.94,0.82,1.227941176,0  ],
    [1.48,1.41,0.37,1.47,0.57,1.69,1.46,0.45,1.5,0.99,0.65,1.157534247,1  ],
    [1.45,1.38,0.36,1.42,0.56,1.6,1.41,0.45,1.44,0.97,0.69,1.134751773,0  ],
    [1.41,1.38,0.33,1.4,0.57,1.6,1.41,0.41,1.2,1,0.75,1.134751773,0  ],
    [1.44,1.36,0.32,1.42,0.61,1.66,1.41,0.44,1.09,0.98,0.76,1.177304965,1  ],
    [1.48,1.34,0.37,1.4,0.58,1.6,1.4,0.47,1.38,0.95,0.71,1.142857143,1  ],
    [1.44,1.38,0.3,1.43,0.47,1.64,1.34,0.42,1.16,0.93,0.81,1.223880597,1  ],
    [1.48,1.32,0.32,1.4,0.57,1.66,1.37,0.46,1.17,0.93,0.85,1.211678832,1  ],
    [1.45,1.33,0.38,1.42,0.56,1.68,1.4,0.46,1.16,0.97,0.81,1.2,0  ],
    [1.43,1.35,0.3,1.38,0.46,1.67,1.38,0.44,1,0.97,0.96,1.210144928,0  ],
    [1.46,1.34,0.37,1.44,0.57,1.61,1.41,0.46,1.33,0.97,0.71,1.141843972,0  ],
    [1.38,1.36,0.27,1.4,0.48,1.61,1.4,0.37,1.25,1.01,0.74,1.15,0  ],
    [1.38,1.33,0.33,1.37,0.55,1.52,1.35,0.41,1,0.98,0.87,1.125925926,0  ],
    [1.32,1.36,0.24,1.37,0.45,1.63,1.37,0.38,1.26,1.04,0.7,1.189781022,0  ],
    [1.4,1.31,0.21,1.35,0.4,1.6,1.35,0.35,1,0.96,0.88,1.185185185,0  ]
  ]

  module.exports = treinox