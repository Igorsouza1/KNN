const todasMedidas = [
  [0.51,0.5,1.33,1.31,0.32,0.55,1.33,0.46,0.18,1.6,0.46,0.45,1.3,0.45,0.75,0.7,0.98,0.98,0.73,0.98,0.98  ],
  [0.53,0.54,1.43,1.37,0.27,0.57,1.42,0.51,0.17,1.66,0.4,0.52,1.41,0.42,0.84,0.53,1.3,0.99,0.68,0.99,1.02  ],
  [0.65,0.6,1.53,1.36,0.42,0.62,1.44,0.58,0.18,1.76,0.54,0.57,1.44,0.51,0.82,0.724137931,1.5,1,0.756097561,0.941176471,0.923076923  ],
  [0.54,0.54,1.34,1.36,0.34,0.56,1.37,0.43,0.16,1.57,0.44,0.46,1.36,0.44,0.8,0.79,1.05,0.99,0.7,1.01,1  ],
  [0.46,0.44,1.37,1.33,0.3,0.59,1.37,0.46,0.17,1.54,0.4,0.49,1.35,0.41,0.76,0.65,1.22,0.99,0.78,0.99,0.96  ],
  [0.54,0.53,1.42,1.36,0.27,0.55,1.4,0.45,0.17,1.55,0.42,0.52,1.38,0.4,0.83,0.6,1.24,0.99,0.66,0.97,0.98  ],
  [0.6,0.54,1.46,1.36,0.35,0.59,1.47,0.58,0.18,1.66,0.45,0.6,1.48,0.46,0.89,0.6,1.33,1.01,0.66,1.01,0.9  ],
  [0.57,0.56,1.45,1.4,0.34,0.6,1.45,0.56,0.18,1.66,0.49,0.58,1.44,0.47,0.84,0.61,1.18,0.99,0.71,0.99,0.98  ],
  [0.52,0.52,1.41,1.38,0.38,0.59,1.44,0.54,0.19,1.59,0.4,0.5,1.41,0.49,0.82,0.7,1.25,0.98,0.72,1,1  ],
  [0.58,0.54,1.36,0.51,0.33,0.6,1.4,0.47,0.17,1.63,0.48,0.51,1.39,0.45,0.79,0.7,1.06,0.99,0.76,1.02,0.93  ],
  [0.5,0.52,1.42,1.34,0.27,0.55,1.36,0.45,0.17,1.65,0.42,0.5,1.37,0.4,0.82,0.6,1.19,1.01,0.67,0.96,1.04  ],
  [0.62,0.58,1.5,1.42,0.36,0.63,1.5,0.55,0.18,1.69,0.49,0.6,1.49,0.45,0.86,0.65,1.22,0.99,0.73,0.99,0.94  ],
  [0.52,0.53,1.4,1.33,0.3,0.57,1.36,0.43,0.17,1.62,0.42,0.54,1.35,0.41,0.78,0.7,1.29,0.99,0.73,0.96,1.02  ],
  [0.57,0.5,1.43,1.36,0.34,0.6,1.4,0.46,0.19,1.74,0.44,0.52,1.42,0.46,0.82,0.74,1.18,1.01,0.73,0.99,0.88  ],
  [0.58,0.54,1.46,1.4,0.36,0.59,1.43,0.47,0.19,1.67,0.44,0.46,1.3,0.42,0.71,0.77,1.05,0.91,0.83,0.89,0.93  ],
  [0.54,0.54,1.39,1.34,0.28,0.53,1.37,0.43,0.17,1.5,0.45,0.47,1.36,0.42,0.83,0.65,1.04,0.99,0.64,0.98,1  ],
  [0.52,0.53,1.43,1.36,0.38,0.6,1.41,0.55,0.17,1.68,0.44,0.6,1.41,0.44,0.81,0.69,1.36,1,0.74,0.99,1.02  ],
  [0.52,0.52,1.43,1.35,0.37,0.69,1.4,0.53,0.17,1.69,0.44,0.59,1.41,0.44,0.72,0.7,1.34,1.01,0.96,0.99,1  ],
  [0.52,0.56,1.4,1.32,0.34,0.6,1.36,0.45,0.17,1.62,0.45,0.54,1.35,0.45,0.75,0.76,1.2,0.99,0.8,0.96,1.08  ],
  [0.48,0.5,1.42,1.32,0.35,0.57,1.39,0.55,0.18,1.63,0.46,0.57,1.37,0.44,0.8,0.64,1.24,0.99,0.71,0.96,1.04  ],
  [0.51,0.53,1.37,1.35,0.3,0.57,1.39,0.46,0.17,1.63,0.42,0.49,1.37,0.42,0.8,0.65,1.17,0.99,0.71,1,1.04  ],
  [0.54,0.55,1.43,1.32,0.37,0.62,1.4,0.59,0.17,1.69,0.48,0.57,1.38,0.48,0.76,0.63,1.19,0.99,0.82,0.97,1.02  ],
  [0.56,0.58,1.47,1.34,0.38,0.63,1.42,0.6,0.18,1.69,0.51,0.61,1.41,0.46,0.78,0.63,1.2,0.99,0.81,0.96,1.04  ],
  [0.61,0.57,1.48,1.38,0.38,0.66,1.47,0.61,0.18,1.61,0.52,0.69,1.45,0.51,0.79,0.62,1.33,0.99,0.84,0.98,0.93  ],
  [0.52,0.53,1.45,1.4,0.34,0.63,1.45,0.53,0.17,1.62,0.39,0.54,1.43,0.4,0.8,0.64,1.38,0.99,0.79,0.99,1.02  ],
  [0.51,0.5,1.38,1.34,0.27,0.58,1.37,0.44,0.17,1.77,0.44,0.5,1.37,0.42,0.79,0.61,1.14,1,0.73,0.99,0.98  ],
  [0.55,0.52,1.4,1.41,0.34,0.57,1.39,0.47,0.2,1.62,0.41,0.4,1.39,0.4,0.82,0.72,0.98,1,0.7,0.99,0.95  ],
  [0.52,0.52,1.45,1.38,0.34,0.53,1.43,0.46,0.18,1.61,0.44,0.55,1.33,0.44,0.8,0.74,1.25,0.93,0.66,0.92,1  ],
  [0.52,0.52,1.44,1.41,0.34,0.63,1.44,0.58,0.18,1.61,0.42,0.52,1.43,0.43,0.8,0.59,1.24,0.99,0.79,0.99,1  ],
  [0.51,0.5,1.47,1.35,0.36,0.63,1.42,0.55,0.18,1.66,0.41,0.6,1.43,0.49,0.8,0.65,1.46,1.01,0.79,0.97,0.98  ],
  [0.5,0.48,1.46,1.29,0.35,0.6,1.39,0.5,0.2,1.64,0.37,0.56,1.36,0.46,0.76,0.7,1.51,0.98,0.79,0.93,0.96  ],
  [0.58,0.54,1.43,1.37,0.36,0.6,1.43,0.54,0.17,1.68,0.48,0.58,1.41,0.47,0.81,0.67,1.21,0.99,0.74,0.99,0.93  ],
  [0.56,0.56,1.45,1.38,0.31,0.61,1.39,0.46,0.18,1.63,0.45,0.5,1.37,0.41,0.76,0.67,1.11,0.99,0.8,0.94,1  ],
  [0.52,0.56,1.4,1.32,0.34,0.6,1.36,0.45,0.17,1.62,0.45,0.54,1.35,0.45,0.75,0.76,1.2,0.99,0.8,0.96,1.08  ],
  [0.57,0.57,1.48,1.38,0.34,0.64,1.44,0.51,0.17,1.57,0.5,0.7,1.42,0.46,0.78,0.67,1.4,0.99,0.82,0.96,1  ],
  [0.55,0.54,1.42,1.34,0.37,0.59,1.4,0.56,0.18,1.58,0.43,0.59,1.39,0.44,0.8,0.66,1.37,0.99,0.74,0.98,0.98  ],
  [0.53,0.52,1.45,1.4,0.34,0.61,1.44,0.53,0.18,1.7,0.43,0.55,1.44,0.42,0.83,0.64,1.28,1,0.73,0.99,0.98  ],
  [0.52,0.51,1.41,1.39,0.32,0.58,1.42,0.54,0.18,1.64,0.4,0.53,1.4,0.41,0.82,0.59,1.32,0.99,0.71,0.99,0.98  ],
  [0.53,0.53,1.39,1.35,0.34,0.55,1.38,0.54,0.17,1.62,0.41,0.54,1.37,0.42,0.82,0.63,1.32,0.99,0.67,0.99,1  ],
  [0.55,0.57,1.4,1.35,0.3,0.57,1.39,0.45,0.18,1.66,0.44,0.54,1.31,0.43,0.74,0.67,1.23,0.94,0.77,0.94,1.04  ],
  [0.58,0.55,1.39,1.31,0.25,0.53,1.37,0.46,0.19,1.58,0.52,0.5,1.36,0.4,0.83,0.54,0.96,0.99,0.64,0.98,0.95  ],
  [0.54,0.53,1.4,1.34,0.32,0.61,1.4,0.56,0.19,1.65,0.41,0.52,1.37,0.41,0.76,0.57,1.27,0.98,0.8,0.98,0.98  ],
  [0.52,0.51,1.47,1.43,0.32,0.63,1.47,0.56,0.18,1.6,0.4,0.52,1.45,0.41,0.82,0.57,1.3,0.99,0.77,0.99,0.98  ],
  [0.57,0.55,1.47,1.44,0.36,0.61,1.47,0.45,0.19,1.61,0.45,0.46,1.33,0.47,0.72,0.8,1.02,0.9,0.85,0.9,0.96  ],
  [0.51,0.52,1.42,1.38,0.33,0.62,1.41,0.53,0.18,1.73,0.41,0.58,1.39,0.4,0.77,0.62,1.41,0.99,0.81,0.98,1.02  ],
  [0.47,0.53,1.31,1.33,0.28,0.54,1.36,0.45,0.17,1.61,0.43,0.5,1.35,0.4,0.81,0.62,1.16,0.99,0.67,1.03,1.13  ],
  [0.54,0.54,1.45,1.42,0.33,0.63,1.45,0.55,0.18,1.86,0.42,0.53,1.43,0.42,0.8,0.6,1.26,0.99,0.79,0.99,1  ],
  [0.55,0.53,1.42,1.35,0.3,0.52,1.39,0.44,0.18,1.6,0.44,0.47,1.37,0.43,0.85,0.68,1.07,0.99,0.61,0.96,0.96  ],
  [0.52,0.53,1.36,1.35,0.28,0.57,1.38,0.45,0.17,1.6,0.42,0.5,1.36,0.44,0.79,0.62,1.19,0.99,0.72,1,1.02  ],
  [0.56,0.55,1.42,1.38,0.36,0.59,1.44,0.54,0.18,1.7,0.43,0.53,1.42,0.44,0.83,0.67,1.23,0.99,0.71,1,0.98  ],
  [0.55,0.53,1.41,1.37,0.32,0.6,1.4,0.45,0.17,1.64,0.44,0.56,1.39,0.46,0.79,0.71,1.27,0.99,0.76,0.99,0.96  ],
  [0.54,0.53,1.44,1.36,0.38,0.6,1.42,0.55,0.18,1.69,0.44,0.59,1.42,0.44,0.82,0.69,1.34,1,0.73,0.99,0.98  ],
  [0.53,0.53,1.41,1.33,0.3,0.57,1.35,0.44,0.18,1.69,0.43,0.52,1.36,0.4,0.79,0.68,1.21,1.01,0.72,0.96,1  ],
  [0.58,0.54,1.39,1.31,0.26,0.58,1.35,0.46,0.17,1.6,0.44,0.46,1.35,0.42,0.77,0.57,1.05,1,0.75,0.97,0.93  ],
  [0.6,0.53,1.37,1.37,0.27,0.57,1.38,0.45,0.17,1.5,0.43,0.48,1.37,0.38,0.8,0.6,1.12,0.99,0.71,1,0.88  ],
  [0.56,0.56,1.51,1.36,0.38,0.65,1.45,0.62,0.18,1.64,0.5,0.67,1.43,0.46,0.78,0.61,1.34,0.99,0.83,0.95,1  ],
  [0.58,0.52,1.44,1.35,0.31,0.61,1.41,0.55,0.18,1.65,0.45,0.56,1.4,0.45,0.79,0.56,1.24,0.99,0.77,0.97,0.9  ],
  [0.53,0.5,1.41,1.41,0.3,0.4,1.45,0.49,0.18,1.68,0.46,0.57,1.45,0.42,1.05,0.61,1.24,1,0.38,1.03,0.94  ],
  [0.54,0.55,1.39,1.31,0.31,0.59,1.39,0.54,0.17,1.46,0.45,0.63,1.36,0.42,0.77,0.57,1.4,0.98,0.77,0.98,1.02  ],
  [0.51,0.5,1.33,1.31,0.32,0.55,1.33,0.46,0.18,1.6,0.46,0.45,1.3,0.45,0.75,0.7,0.98,0.98,0.73,0.98,0.98  ],
  [0.56,0.54,1.41,1.36,0.32,0.6,1.4,0.54,0.17,1.59,0.4,0.56,1.38,0.42,0.78,0.59,1.4,0.99,0.77,0.98,0.96  ],
  [0.52,0.46,1.43,1.37,0.31,0.57,1.42,0.52,0.19,1.72,0.4,0.55,1.41,0.45,0.84,0.6,1.38,0.99,0.68,0.99,0.88  ],
  [0.59,0.54,1.48,1.41,0.36,0.6,1.42,0.58,0.18,1.63,0.43,0.61,1.42,0.45,0.82,0.62,1.42,1,0.73,0.96,0.92  ],
  [0.55,0.55,1.48,1.31,0.36,0.6,1.39,0.58,0.17,1.59,0.49,0.64,1.37,0.45,0.77,0.62,1.31,0.99,0.78,0.93,1  ],
  [0.56,0.54,1.43,1.4,0.32,0.6,1.44,0.56,0.18,1.65,0.44,0.55,1.41,0.43,0.81,0.57,1.25,0.98,0.74,0.99,0.96  ],
  [0.54,0.53,1.35,1.33,0.33,0.56,1.34,0.44,0.17,1.48,0.43,0.48,1.33,0.42,0.77,0.75,1.12,0.99,0.73,0.99,0.98  ],
  [0.56,0.57,1.44,1.34,0.38,0.65,1.43,0.69,0.18,1.6,0.46,0.65,1.4,0.48,0.75,0.55,1.41,0.98,0.87,0.97,1.02  ],
  [0.53,0.56,1.4,1.31,0.33,0.59,1.41,0.56,0.17,1.58,0.47,0.64,1.37,0.46,0.78,0.59,1.36,0.97,0.76,0.98,1.06  ],
  [0.56,0.54,1.43,1.35,0.32,0.57,1.41,0.54,0.18,1.58,0.42,0.51,1.42,0.42,0.85,0.59,1.21,1.01,0.67,0.99,0.96  ],
  [0.6,0.56,1.45,1.38,0.39,0.61,1.45,0.48,0.18,1.61,0.46,0.68,1.43,0.48,0.82,0.81,1.48,0.99,0.74,0.99,0.93  ],
  [0.54,0.53,1.41,1.38,0.28,0.57,1.39,0.44,0.17,1.6,0.45,0.51,1.38,0.4,0.81,0.64,1.13,0.99,0.7,0.98,0.98  ],
  [0.55,0.55,1.44,1.42,0.37,0.62,1.44,0.49,0.18,1.64,0.48,0.41,1.32,0.46,0.7,0.76,0.85,0.92,0.89,0.92,1  ],
  [0.5,0.48,1.41,1.36,0.36,0.58,1.4,0.56,0.17,1.64,0.4,0.54,1.38,0.4,0.8,0.64,1.35,0.99,0.72,0.98,0.96  ],
  [0.53,0.53,1.45,1.4,0.35,0.53,1.45,0.57,0.18,1.67,0.42,0.55,1.43,0.41,0.9,0.61,1.31,0.99,0.59,0.99,1  ],
  [0.58,0.49,1.38,1.33,0.26,0.56,1.37,0.43,0.17,1.61,0.43,0.52,1.35,0.42,0.79,0.6,1.21,0.99,0.71,0.98,0.84  ],
  [0.56,0.56,1.48,1.42,0.36,0.57,1.49,0.49,0.2,1.69,0.44,0.48,1.49,0.47,0.92,0.73,1.09,1,0.62,1.01,1  ],
  [0.48,0.54,1.38,1.35,0.29,0.58,1.38,0.45,0.17,1.66,0.43,0.55,1.36,0.4,0.78,0.64,1.28,0.99,0.74,0.99,1.13  ],
  [0.55,0.57,1.33,1.29,0.32,0.57,1.35,0.48,0.18,1.51,0.45,0.5,1.31,0.42,0.74,0.67,1.11,0.97,0.77,0.98,1.04  ],
  [0.53,0.5,1.41,1.41,0.3,0.4,1.45,0.49,0.18,1.68,0.46,0.57,1.45,0.42,1.05,0.61,1.24,1,0.38,1.03,0.94  ],
  [0.55,0.52,1.43,1.31,0.3,0.54,1.35,0.45,0.18,1.62,0.45,0.45,1.35,0.41,0.81,0.67,1,1,0.67,0.94,0.95  ],
  [0.54,0.55,1.42,1.36,0.29,0.55,1.39,0.48,0.19,1.69,0.44,0.53,1.4,0.4,0.85,0.6,1.2,1.01,0.65,0.99,1.02  ],
  [0.57,0.53,1.3,1.32,0.29,0.59,1.35,0.45,0.17,1.66,0.43,0.55,1.35,0.42,0.76,0.64,1.28,1,0.78,1.04,0.93  ],
  [0.5,0.5,1.44,1.34,0.37,0.63,1.4,0.5,0.19,1.61,0.42,0.55,1.4,0.39,0.77,0.74,1.31,1,0.82,0.97,1  ],
  [0.52,0.52,1.41,1.38,0.37,0.61,1.42,0.54,0.19,1.77,0.4,0.5,1.41,0.41,0.8,0.69,1.25,0.99,0.76,1,1  ],
  [0.59,0.59,1.44,1.43,0.36,0.61,1.49,0.49,0.2,1.72,0.45,0.49,1.49,0.47,0.88,0.73,1.09,1,0.69,1.03,1  ],
  [0.55,0.55,1.45,1.37,0.34,0.55,1.4,0.42,0.18,1.5,0.42,0.52,1.38,0.46,0.83,0.81,1.24,0.99,0.66,0.95,1  ],
  [0.59,0.56,1.46,1.4,0.36,0.59,1.43,0.59,0.18,1.71,0.42,0.5,1.44,0.44,0.85,0.61,1.19,1.01,0.69,0.99,0.95  ],
  [0.59,0.53,1.4,1.32,0.26,0.65,1.37,0.47,0.18,1.68,0.47,0.45,1.36,0.45,0.71,0.55,0.96,0.99,0.92,0.97,0.9  ],
  [0.56,0.51,1.46,1.37,0.33,0.64,1.43,0.54,0.17,1.75,0.42,0.53,1.41,0.46,0.77,0.61,1.26,0.99,0.83,0.97,0.91  ],
  [0.58,0.54,1.43,1.34,0.35,0.62,1.42,0.54,0.17,1.58,0.45,0.61,1.41,0.48,0.79,0.65,1.36,0.99,0.78,0.99,0.93  ],
  [0.56,0.57,1.45,1.37,0.33,0.58,1.36,0.47,0.18,1.66,0.42,0.46,1.36,0.46,0.78,0.7,1.1,1,0.74,0.94,1.02  ],
  [0.56,0.5,1.45,1.37,0.39,0.61,1.43,0.52,0.18,1.74,0.46,0.56,1.41,0.47,0.8,0.75,1.22,0.99,0.76,0.97,0.89  ],
  [0.56,0.57,1.42,1.41,0.36,0.63,1.38,0.57,0.18,1.6,0.45,0.63,1.4,0.43,0.77,0.63,1.4,1.01,0.82,0.99,1.02  ],
  [0.57,0.55,1.4,1.34,0.31,0.55,1.36,0.47,0.18,1.6,0.47,0.47,1.36,0.45,0.81,0.66,1,1,0.68,0.97,0.96  ],
  [0.54,0.55,1.45,1.41,0.33,0.57,0.46,0.48,0.18,1.6,0.46,0.59,1.38,0.43,0.81,0.69,1.28,3,0.7,0.95,1.02  ],
  [0.55,0.56,1.48,1.44,0.35,0.59,1.49,0.56,0.17,1.69,0.43,0.57,1.46,0.42,0.87,0.62,1.33,0.98,0.68,0.99,1.02  ],
  [0.6,0.55,1.44,0.48,0.33,0.6,1.39,0.45,0.17,1.64,0.45,0.48,1.35,0.46,0.75,0.73,1.07,0.97,0.8,0.94,0.92  ],
  [0.55,0.55,1.4,1.31,0.34,0.61,1.4,0.54,0.17,1.5,0.45,0.66,1.37,0.47,0.76,0.63,1.47,0.98,0.8,0.98,1  ],
  [0.54,0.55,1.42,1.36,0.33,0.59,1.45,0.55,0.21,1.77,0.44,0.52,1.42,0.46,0.83,0.6,1.18,0.98,0.71,1,1.02  ],
  [0.46,0.46,1.38,1.34,0.3,0.54,1.38,0.46,0.17,1.56,0.38,0.48,1.36,0.39,0.82,0.65,1.26,0.99,0.66,0.99,1  ],
  [0.56,0.56,1.5,1.4,0.38,0.51,1.41,0.48,0.17,1.68,0.47,0.54,1.4,0.5,0.89,0.79,1.15,0.99,0.57,0.93,1  ],
  [0.55,0.5,1.38,1.33,0.25,0.5,1.36,0.42,0.17,1.5,0.42,0.48,1.35,0.37,0.85,0.6,1.14,0.99,0.59,0.98,0.91  ],
  [0.51,0.49,1.46,1.4,0.37,0.58,1.44,0.56,0.19,1.76,0.43,0.52,1.43,0.43,0.85,0.66,1.21,0.99,0.68,0.98,0.96  ],
  [0.48,0.48,1.37,1.35,0.3,0.56,1.38,0.47,0.17,1.57,0.4,0.48,1.37,0.39,0.81,0.64,1.2,0.99,0.69,1,1  ],
  [0.59,0.56,1.43,1.35,0.31,0.56,1.42,0.47,0.18,1.62,0.48,0.49,1.39,0.48,0.83,0.66,1.02,0.98,0.67,0.97,0.95  ],
  [0.52,0.47,1.5,1.4,0.34,0.58,1.4,0.47,0.18,1.61,0.43,0.6,1.39,0.39,0.81,0.72,1.4,0.99,0.72,0.93,0.9  ],
  [0.56,0.55,1.4,1.3,0.31,0.62,1.38,0.57,0.17,1.55,0.49,0.66,1.36,0.45,0.74,0.54,1.35,0.99,0.84,0.97,0.98  ],
  [0.61,0.57,1.49,1.41,0.36,0.6,1.42,0.5,0.18,1.55,0.52,0.55,1.41,0.5,0.81,0.72,1.06,0.99,0.74,0.95,0.93  ],
  [0.57,0.58,1.43,1.32,0.33,0.62,1.38,0.57,0.17,1.48,0.44,0.67,1.36,0.44,0.74,0.58,1.52,0.99,0.84,0.95,1.02  ],
  [0.58,0.58,1.49,1.39,0.39,0.6,1.4,0.55,0.17,1.62,0.48,0.59,1.38,0.46,0.78,0.71,1.23,0.99,0.77,0.93,1  ],
  [0.57,0.58,1.48,1.32,0.33,0.65,1.41,0.59,0.18,1.59,0.49,0.7,1.41,0.49,0.76,0.56,1.43,1,0.86,0.95,1.02  ],
  [0.63,0.54,1.43,1.35,0.38,0.61,1.43,0.58,0.17,1.6,0.48,0.7,1.42,0.47,0.81,0.66,1.46,0.99,0.75,0.99,0.86  ],
  [0.57,0.55,1.43,1.35,0.32,0.62,1.42,0.59,0.18,1.56,0.51,0.66,1.4,0.46,0.78,0.54,1.29,0.99,0.79,0.98,0.96  ],
  [0.5,0.5,1.43,1.36,0.32,0.61,1.41,0.53,0.2,1.6,0.43,0.52,1.4,0.41,0.79,0.6,1.21,0.99,0.77,0.98,1  ],
  [0.51,0.51,1.38,1.36,0.32,0.56,1.41,0.44,0.18,1.64,0.44,0.49,1.37,0.4,0.81,0.73,1.11,0.97,0.69,0.99,1  ],
  [0.57,0.54,1.55,1.34,0.4,0.62,1.44,0.51,0.18,1.77,0.54,0.6,1.43,0.54,0.81,0.784313725,1.1,0.993055556,0.765432099,0.922580645,0.947368421  ],
  [0.56,0.54,1.38,1.39,0.34,0.57,1.42,0.46,0.19,1.69,0.42,0.43,1.38,0.45,0.81,0.74,1.02,0.97,0.7,1,0.96  ],
  [0.58,0.55,1.4,1.33,0.33,0.55,1.35,0.45,0.18,1.68,0.5,0.42,1.35,0.45,0.8,0.73,0.84,1,0.69,0.96,0.95  ],
  [0.54,0.52,1.4,1.33,0.37,0.59,1.38,0.55,0.17,1.68,0.43,0.59,1.38,0.43,0.79,0.67,1.37,1,0.75,0.99,0.96  ],
  [0.55,0.53,1.44,1.38,0.36,0.6,1.43,0.53,0.18,1.65,0.42,0.54,1.41,0.49,0.81,0.68,1.29,0.99,0.74,0.98,0.96  ],
  [0.55,0.55,1.41,1.36,0.3,0.58,1.42,0.58,0.18,1.77,0.43,0.55,1.4,0.46,0.82,0.52,1.28,0.99,0.71,0.99,1  ],
  [0.55,0.48,1.47,1.34,0.32,0.59,1.42,0.56,0.21,1.57,0.48,0.54,1.39,0.43,0.8,0.57,1.13,0.98,0.74,0.95,0.87  ],
  [0.53,0.51,1.46,1.4,0.37,0.6,1.45,0.55,0.18,1.73,0.41,0.51,1.43,0.41,0.83,0.67,1.24,0.99,0.72,0.98,0.96  ],
  [0.55,0.56,1.52,1.45,0.38,0.67,1.48,0.61,0.18,1.8,0.46,0.56,1.47,0.45,0.8,0.62,1.22,0.99,0.84,0.97,1.02  ],
  [0.54,0.55,1.43,1.37,0.36,0.59,1.42,0.54,0.18,1.7,0.42,0.53,1.41,0.42,0.82,0.67,1.26,0.99,0.72,0.99,1.02  ],
  [0.55,0.53,1.4,1.34,0.34,0.58,1.39,0.46,0.19,1.67,0.41,0.46,1.31,0.45,0.73,0.74,1.12,0.94,0.79,0.94,0.96  ],
  [0.58,0.56,1.43,1.39,0.26,0.6,1.42,0.45,0.17,1.69,0.42,0.52,1.4,0.39,0.8,0.58,1.24,0.99,0.75,0.98,0.97  ],
  [0.55,0.54,1.43,1.37,0.35,0.57,1.42,0.55,0.19,1.71,0.46,0.58,1.4,0.47,0.83,0.64,1.26,0.99,0.69,0.98,0.98  ],
  [0.54,0.53,1.43,1.37,0.31,0.62,1.41,0.54,0.17,1.52,0.4,0.53,1.4,0.4,0.78,0.57,1.32,0.99,0.79,0.98,0.98  ],
  [0.52,0.52,1.45,1.39,0.32,0.65,1.43,0.61,0.18,1.55,0.44,0.54,1.42,0.44,0.77,0.52,1.23,0.99,0.84,0.98,1  ],
  [0.5,0.51,1.38,1.3,0.31,0.59,1.39,0.51,0.17,1.52,0.44,0.55,1.36,0.43,0.77,0.61,1.25,0.98,0.77,0.99,1.02  ],
  [0.58,0.56,1.45,1.37,0.37,0.65,1.45,0.59,0.18,1.72,0.47,0.59,1.43,0.46,0.78,0.63,1.26,0.99,0.83,0.99,0.97  ],
  [0.62,0.57,1.48,1.43,0.35,0.62,1.46,0.52,0.18,1.77,0.49,0.62,1.46,0.46,0.84,0.67,1.27,1,0.74,0.99,0.92  ],
  [0.54,0.55,1.39,1.37,0.31,0.59,1.4,0.49,0.18,1.6,0.45,0.44,1.34,0.42,0.75,0.63,0.98,0.96,0.79,0.96,1.02  ],
  [0.51,0.52,1.39,1.36,0.32,0.56,1.39,0.53,0.17,1.58,0.41,0.5,1.38,0.4,0.82,0.6,1.22,0.99,0.68,0.99,1.02  ],
  [0.6,0.56,1.5,1.46,0.38,0.65,1.52,0.56,0.18,1.64,0.45,0.65,1.52,0.45,0.87,0.68,1.44,1,0.75,1.01,0.93  ],
  [0.55,0.53,1.43,1.37,0.31,0.62,1.41,0.58,0.18,1.65,0.38,0.53,1.4,0.39,0.78,0.53,1.39,0.99,0.79,0.98,0.96  ],
  [0.5,0.52,1.38,1.33,0.32,0.6,1.38,0.56,0.17,1.53,0.43,0.52,1.35,0.42,0.75,0.57,1.21,0.98,0.8,0.98,1.04  ],
  [0.58,0.53,1.45,1.39,0.38,0.6,1.43,0.58,0.18,1.68,0.46,0.6,1.43,0.45,0.83,0.66,1.3,1,0.72,0.99,0.91  ],
  [0.53,0.52,1.48,1.34,0.35,0.63,1.42,0.55,0.2,1.73,0.4,0.55,1.41,0.47,0.78,0.64,1.38,0.99,0.81,0.95,0.98  ],
  [0.57,0.54,1.45,1.34,0.4,0.65,1.41,0.57,0.18,1.67,0.49,0.57,1.41,0.47,0.76,0.7,1.16,1,0.86,0.97,0.95  ],
  [0.55,0.52,1.42,1.37,0.27,0.55,1.39,0.42,0.17,1.58,0.42,0.52,1.38,0.39,0.83,0.64,1.24,0.99,0.66,0.97,0.95  ],
  [0.54,0.55,1.39,1.29,0.35,0.61,1.36,0.55,0.17,1.59,0.47,0.65,1.35,0.49,0.74,0.64,1.38,0.99,0.82,0.97,1.02  ],
  [0.55,0.52,1.43,1.33,0.28,0.55,1.37,0.46,0.17,1.6,0.43,0.44,1.37,0.4,0.82,0.61,1.02,1,0.67,0.96,0.95  ],
  [0.55,0.54,1.4,1.43,0.33,0.63,1.43,0.47,0.18,1.73,0.48,0.57,1.39,0.43,0.76,0.7,1.19,0.97,0.83,0.99,0.98  ],
  [0.51,0.53,1.46,1.32,0.33,0.59,1.4,0.57,0.17,1.6,0.43,0.53,1.38,0.46,0.79,0.58,1.23,0.99,0.75,0.95,1.04  ],
  [0.49,0.52,1.37,1.33,0.28,0.53,1.34,0.43,0.17,1.59,0.42,0.49,1.35,0.4,0.82,0.65,1.17,1.01,0.65,0.99,1.06  ],
  [0.56,0.56,1.42,1.36,0.35,0.62,1.44,0.6,0.18,1.6,0.44,0.64,1.42,0.45,0.8,0.58,1.45,0.99,0.78,1,1  ],
  [0.54,0.53,1.4,1.31,0.31,0.59,1.28,0.4,0.17,1.51,0.41,0.5,1.31,0.45,0.72,0.78,1.22,1.02,0.82,0.94,0.98  ],
  [0.51,0.49,1.42,1.27,0.34,0.58,1.39,0.55,0.19,1.59,0.41,0.58,1.35,0.47,0.77,0.62,1.41,0.97,0.75,0.95,0.96  ],
  [0.54,0.54,1.35,1.35,0.34,0.55,1.37,0.45,0.17,1.5,0.46,0.52,1.37,0.46,0.82,0.76,1.13,1,0.67,1.01,1  ],
  [0.57,0.56,1.47,1.39,0.32,0.61,1.44,0.48,0.18,1.68,0.46,0.56,1.43,0.42,0.82,0.67,1.22,0.99,0.74,0.97,0.98  ],
  [0.57,0.54,1.38,1.33,0.27,0.51,1.37,0.44,0.17,1.56,0.43,0.46,1.37,0.38,0.86,0.61,1.07,1,0.59,0.99,0.95  ],
  [0.58,0.53,1.45,1.39,0.38,0.6,1.43,0.58,0.18,1.68,0.46,0.6,1.43,0.45,0.83,0.66,1.3,1,0.72,0.99,0.91  ],
  [0.53,0.53,1.39,1.36,0.3,0.63,1.39,0.62,0.18,1.7,0.42,0.49,1.37,0.42,0.74,0.48,1.17,0.99,0.85,0.99,1  ],
  [0.53,0.5,1.42,1.33,0.3,0.6,1.34,0.45,0.17,1.61,0.43,0.5,1.35,0.38,0.75,0.67,1.16,1.01,0.8,0.95,0.94  ],
  [0.51,0.54,1.38,1.35,0.31,0.56,1.38,0.44,0.18,1.62,0.4,0.51,1.36,0.41,0.8,0.7,1.27,0.99,0.7,0.99,1.06  ],
  [0.55,0.56,1.5,1.44,0.34,0.61,1.47,0.46,0.19,1.72,0.46,0.59,1.43,0.45,0.82,0.74,1.28,0.97,0.74,0.95,1.02  ],
  [0.53,0.54,1.41,1.34,0.33,0.61,1.38,0.57,0.17,1.65,0.45,0.54,1.37,0.43,0.76,0.58,1.2,0.99,0.8,0.97,1.02  ],
  [0.6,0.49,1.43,1.35,0.33,0.59,1.39,0.51,0.18,1.64,0.47,0.61,1.39,0.41,0.8,0.65,1.3,1,0.74,0.97,0.82  ],
  [0.54,0.55,1.49,1.4,0.35,0.6,1.46,0.44,0.18,1.73,0.49,0.6,1.45,0.45,0.85,0.8,1.22,0.99,0.71,0.97,1.02  ],
  [0.55,0.58,1.47,1.4,0.38,0.65,1.4,0.49,0.21,1.66,0.48,0.47,1.39,0.48,0.74,0.78,0.98,0.99,0.88,0.95,1.05  ],
  [0.57,0.57,1.48,1.38,0.35,0.64,1.46,0.6,0.18,1.65,0.5,0.66,1.45,0.46,0.81,0.58,1.32,0.99,0.79,0.98,1  ],
  [0.53,0.54,1.4,1.34,0.33,0.57,1.35,0.5,0.17,1.57,0.46,0.51,1.3,0.45,0.73,0.66,1.11,0.96,0.78,0.93,1.02  ],
  [0.58,0.56,1.42,1.35,0.32,0.59,1.34,0.55,0.17,1.6,0.49,0.61,1.32,0.46,0.73,0.58,1.24,0.99,0.81,0.93,0.97  ],
  [0.53,0.53,1.4,1.3,0.33,0.53,1.31,0.43,0.17,1.73,0.43,0.46,1.31,0.46,0.78,0.77,1.07,1,0.68,0.94,1  ],
  [0.58,0.56,1.45,1.36,0.33,0.58,1.36,0.45,0.18,1.53,0.45,0.51,1.35,0.46,0.77,0.73,1.13,0.99,0.75,0.93,0.97  ],
  [0.54,0.53,1.4,1.33,0.32,0.58,1.37,0.54,0.18,1.5,0.41,0.55,1.36,0.44,0.78,0.59,1.34,0.99,0.74,0.97,0.98  ],
  [0.57,0.56,1.47,1.39,0.33,0.56,1.42,0.49,0.18,1.63,0.45,0.53,1.33,0.43,0.77,0.67,1.18,0.94,0.73,0.9,0.98  ],
  [0.52,0.51,1.39,1.33,0.33,0.64,1.38,0.57,0.17,1.7,0.42,0.54,1.35,0.43,0.71,0.58,1.29,0.98,0.9,0.97,0.98  ],
  [0.55,0.54,1.45,1.4,0.36,0.67,1.45,0.64,0.18,1.88,0.42,0.55,1.43,0.43,0.76,0.56,1.31,0.99,0.88,0.99,0.98  ],
  [0.54,0.56,1.41,1.37,0.32,0.59,1.42,0.46,0.18,1.68,0.43,0.5,1.39,0.42,0.8,0.7,1.16,0.98,0.74,0.99,1.04  ],
  [0.55,0.55,1.42,1.36,0.34,0.59,1.37,0.48,0.18,1.6,0.45,0.53,1.36,0.42,0.77,0.71,1.18,0.99,0.77,0.96,1  ],
  [0.55,0.56,1.4,1.37,0.41,0.61,1.39,0.56,0.17,1.58,0.5,0.63,1.37,0.48,0.76,0.73,1.26,0.99,0.8,0.98,1.02  ],
  [0.57,0.51,1.42,1.36,0.34,0.55,1.39,0.46,0.17,1.65,0.45,0.5,1.34,0.46,0.79,0.74,1.11,0.96,0.7,0.94,0.89  ],
  [0.54,0.57,1.4,1.3,0.33,0.61,1.39,0.57,0.17,1.47,0.46,0.64,1.37,0.44,0.76,0.58,1.39,0.99,0.8,0.98,1.06  ],
  [0.53,0.52,1.4,1.36,0.33,0.6,1.4,0.55,0.18,1.75,0.42,0.52,1.37,0.42,0.77,0.6,1.24,0.98,0.78,0.98,0.98  ],
  [0.53,0.54,1.43,1.33,0.37,0.67,1.42,0.57,0.17,1.72,0.47,0.56,1.41,0.46,0.74,0.65,1.19,0.99,0.91,0.99,1.02  ],
  [0.57,0.55,1.46,1.38,0.35,0.59,1.44,0.59,0.17,1.6,0.52,0.72,1.43,0.52,0.84,0.59,1.38,0.99,0.7,0.98,0.96  ],
  [0.56,0.56,1.45,1.42,0.36,0.61,1.44,0.47,0.2,1.64,0.45,0.46,1.32,0.46,0.71,0.77,1.02,0.92,0.86,0.91,1  ],
  [0.55,0.55,1.42,1.36,0.34,0.63,1.43,0.59,0.17,1.57,0.46,0.66,1.42,0.46,0.79,0.58,1.43,0.99,0.8,1,1  ],
  [0.6,0.56,1.45,1.36,0.36,0.63,1.48,0.59,0.18,1.61,0.46,0.58,1.46,0.36,0.83,0.61,1.26,0.99,0.76,1.01,0.93  ],
  [0.55,0.56,1.52,1.45,0.38,0.67,1.48,0.61,0.18,1.8,0.46,0.56,1.47,0.45,0.8,0.62,1.22,0.99,0.84,0.97,1.02  ],
  [0.59,0.55,1.43,1.32,0.3,0.56,1.37,0.45,0.18,1.66,0.48,0.43,1.36,0.44,0.8,0.67,0.9,0.99,0.7,0.95,0.93  ],
  [0.59,0.57,1.48,1.41,0.36,0.59,1.49,0.56,0.17,1.7,0.47,0.59,1.49,0.45,0.9,0.64,1.26,1,0.66,1.01,0.97  ],
  [0.5,0.49,1.42,1.34,0.32,0.59,1.37,0.57,0.17,1.52,0.42,0.56,1.36,0.41,0.77,0.56,1.33,0.99,0.77,0.96,0.98  ],
  [0.53,0.51,1.41,1.35,0.35,0.61,1.4,0.55,0.19,1.6,0.39,0.5,1.41,0.44,0.8,0.64,1.28,1.01,0.76,1,0.96  ],
  [0.49,0.49,1.35,1.32,0.31,0.6,1.33,0.51,0.17,1.57,0.45,0.5,1.31,0.46,0.71,0.61,1.11,0.98,0.85,0.97,1  ],
  [0.51,0.53,1.45,1.38,0.33,0.57,1.41,0.43,0.18,1.57,0.45,0.44,1.34,0.42,0.77,0.77,0.98,0.95,0.74,0.92,1.04  ],
  [0.57,0.56,1.39,1.38,0.33,0.59,1.45,0.54,0.17,1.68,0.47,0.53,1.44,0.44,0.85,0.61,1.13,0.99,0.69,1.04,0.98  ],
  [0.57,0.53,1.45,1.3,0.26,0.56,1.35,0.46,0.18,1.64,0.46,0.46,1.35,0.42,0.79,0.57,1,1,0.71,0.93,0.93  ],
  [0.58,0.57,1.5,1.34,0.32,0.62,1.41,0.59,0.17,1.6,0.57,0.6,1.39,0.45,0.77,0.54,1.05,0.99,0.81,0.93,0.98  ],
  [0.48,0.44,1.37,1.34,0.3,0.6,1.39,0.48,0.17,1.59,0.41,0.52,1.37,0.4,0.77,0.62,1.27,0.99,0.78,1,0.92  ],
  [0.54,0.52,1.52,1.48,0.36,0.63,1.52,0.57,0.18,1.8,0.42,0.51,1.5,0.43,0.87,0.63,1.21,0.99,0.72,0.99,0.96  ],
  [0.56,0.55,1.35,1.3,0.3,0.55,1.37,0.45,0.18,1.62,0.47,0.48,1.35,0.42,0.8,0.67,1.02,0.99,0.69,1,0.98  ],
  [0.59,0.58,1.47,1.38,0.43,0.59,1.44,0.59,0.18,1.59,0.55,0.68,1.44,0.54,0.85,0.73,1.24,1,0.69,0.98,0.98  ],
  [0.55,0.53,1.47,1.41,0.3,0.57,1.43,0.54,0.18,1.6,0.43,0.53,1.45,0.42,0.88,0.56,1.23,1.01,0.65,0.99,0.96  ],
  [0.61,0.54,1.49,1.38,0.39,0.68,1.41,0.61,0.18,1.56,0.44,0.73,1.4,0.46,0.72,0.64,1.66,0.99,0.94,0.94,0.89  ],
  [0.52,0.56,1.4,1.36,0.33,0.56,1.39,0.44,0.17,1.66,0.44,0.53,1.38,0.44,0.82,0.75,1.2,0.99,0.68,0.99,1.08  ],
  [0.57,0.57,1.4,1.33,0.28,0.6,1.37,0.47,0.18,1.7,0.47,0.44,1.37,0.47,0.77,0.6,0.94,1,0.78,0.98,1  ],
  [0.58,0.49,1.38,1.33,0.26,0.56,1.37,0.43,0.17,1.61,0.43,0.52,1.35,0.42,0.79,0.6,1.21,0.99,0.71,0.98,0.84  ],
  [0.55,0.54,1.44,1.38,0.36,0.58,1.43,0.48,0.17,1.68,0.48,0.54,1.43,0.47,0.85,0.75,1.13,1,0.68,0.99,0.98  ],
  [0.51,0.49,1.4,1.35,0.27,0.56,1.37,0.43,0.17,1.61,0.45,0.53,1.38,0.38,0.82,0.63,1.18,1.01,0.68,0.99,0.96  ],
  [0.51,0.51,1.35,1.33,0.31,0.54,1.38,0.44,0.16,1.58,0.42,0.5,1.36,0.41,0.82,0.7,1.19,0.99,0.66,1.01,1  ],
  [0.55,0.57,1.48,1.36,0.27,0.57,1.38,0.54,0.18,1.71,0.42,0.53,1.4,0.46,0.83,0.5,1.26,1.01,0.69,0.95,1.04  ],
  [0.54,0.48,1.35,1.29,0.3,0.54,1.35,0.43,0.17,1.6,0.41,0.52,1.35,0.39,0.81,0.7,1.27,1,0.67,1,0.89  ],
  [0.6,0.57,1.44,1.38,0.35,0.6,1.37,0.58,0.17,1.67,0.47,0.57,1.36,0.47,0.76,0.6,1.21,0.99,0.79,0.94,0.95  ],
  [0.55,0.54,1.48,1.41,0.37,0.69,1.47,0.57,0.18,1.69,0.46,0.69,1.46,0.45,0.77,0.65,1.5,0.99,0.9,0.99,0.98  ],
  [0.55,0.56,1.45,1.38,0.36,0.63,1.42,0.56,0.18,1.6,0.45,0.65,1.41,0.45,0.78,0.64,1.44,0.99,0.81,0.97,1.02  ],
  [0.55,0.56,1.41,1.38,0.33,0.54,1.4,0.57,0.17,1.6,0.46,0.55,1.41,0.41,0.87,0.58,1.2,1.01,0.62,1,1.02  ],
  [0.47,0.55,1.44,1.36,0.32,0.61,1.42,0.61,0.18,1.66,0.53,0.58,1.41,0.44,0.8,0.52,1.09,0.99,0.76,0.98,1.17  ],
  [0.58,0.57,1.48,1.34,0.37,0.63,1.4,0.58,0.17,1.6,0.48,0.66,1.4,0.47,0.77,0.64,1.38,1,0.82,0.95,0.98  ],
  [0.56,0.56,1.44,1.38,0.3,0.59,1.43,0.47,0.17,1.64,0.45,0.52,1.34,0.42,0.75,0.64,1.16,0.94,0.79,0.93,1  ],
  [0.42,0.55,1.48,1.32,0.32,0.59,1.4,0.57,0.17,1.66,0.46,0.54,1.37,0.46,0.78,0.56,1.17,0.98,0.76,0.93,1.31  ],
  [0.57,0.54,1.45,1.33,0.38,0.61,1.42,0.56,0.18,1.68,0.49,0.57,1.4,0.46,0.79,0.68,1.16,0.99,0.77,0.97,0.95  ],
  [0.58,0.55,1.43,1.35,0.3,0.55,1.38,0.46,0.18,1.67,0.46,0.46,1.38,0.44,0.83,0.65,1,1,0.66,0.97,0.95  ],
  [0.58,0.57,1.46,1.34,0.37,0.64,1.44,0.57,0.18,1.61,0.49,0.65,1.41,0.46,0.77,0.65,1.33,0.98,0.83,0.97,0.98  ],
  [0.62,0.56,1.38,1.36,0.27,0.59,1.4,0.48,0.17,1.61,0.4,0.5,1.4,0.37,0.81,0.56,1.25,1,0.73,1.01,0.9  ],
  [0.54,0.54,1.38,1.33,0.33,0.57,1.37,0.55,0.17,1.52,0.47,0.47,1.35,0.41,0.78,0.6,1,0.99,0.73,0.98,1  ],
  [0.53,0.53,1.32,1.36,0.24,0.56,1.37,0.45,0.17,1.63,0.43,0.54,1.37,0.38,0.81,0.53,1.26,1,0.69,1.04,1  ],
  [0.49,0.46,1.4,1.31,0.21,0.49,1.35,0.4,0.18,1.6,0.4,0.4,1.35,0.35,0.86,0.53,1,1,0.57,0.96,0.94  ],
  [0.57,0.59,1.48,1.39,0.34,0.65,1.46,0.6,0.18,1.6,0.5,0.64,1.45,0.43,0.8,0.57,1.28,0.99,0.81,0.98,1.04  ],
  [0.55,0.54,1.44,1.34,0.32,0.6,1.39,0.56,0.18,1.71,0.42,0.49,1.4,0.43,0.8,0.57,1.17,1.01,0.75,0.97,0.98  ],
  [0.57,0.5,1.4,1.34,0.36,0.5,1.41,0.47,0.18,1.69,0.43,0.6,1.38,0.49,0.88,0.77,1.4,0.98,0.57,0.99,0.88  ],
  [0.5,0.49,1.47,1.34,0.3,0.58,1.37,0.51,0.17,1.6,0.43,0.51,1.35,0.43,0.77,0.59,1.19,0.99,0.75,0.92,0.98  ],
  [0.59,0.51,1.42,1.38,0.32,0.55,1.41,0.46,0.18,1.5,0.41,0.25,1.33,0.46,0.78,0.7,0.61,0.94,0.71,0.94,0.86  ],
  [0.57,0.55,1.45,1.32,0.3,0.58,1.36,0.5,0.18,1.65,0.46,0.48,1.36,0.42,0.78,0.6,1.04,1,0.74,0.94,0.96  ],
  [0.59,0.51,1.45,1.35,0.35,0.61,1.39,0.53,0.18,1.63,0.45,0.55,1.39,0.45,0.78,0.66,1.22,1,0.78,0.96,0.86  ],
  [0.57,0.55,1.47,1.44,0.38,0.67,1.48,0.57,0.18,1.82,0.45,0.52,1.46,0.44,0.79,0.67,1.16,0.99,0.85,0.99,0.96  ],
  [0.61,0.58,1.47,1.36,0.39,0.6,1.44,0.59,0.18,1.59,0.5,0.7,1.44,0.47,0.84,0.66,1.4,1,0.71,0.98,0.95  ],
  [0.57,0.54,1.43,1.37,0.3,0.6,1.4,0.46,0.17,1.68,0.46,0.54,1.4,0.48,0.8,0.65,1.17,1,0.75,0.98,0.95  ],
  [0.55,0.56,1.4,1.32,0.37,0.59,1.39,0.58,0.17,1.6,0.48,0.65,1.37,0.45,0.78,0.64,1.35,0.99,0.76,0.98,1.02  ],
  [0.53,0.52,1.47,1.37,0.38,0.66,1.47,0.57,0.18,1.62,0.46,0.6,1.44,0.47,0.78,0.67,1.3,0.98,0.85,0.98,0.98  ],
  [0.53,0.46,1.43,1.34,0.34,0.59,1.42,0.55,0.21,1.76,0.41,0.53,1.42,0.46,0.83,0.62,1.29,1,0.71,0.99,0.87  ],
  [0.51,0.55,1.42,1.31,0.32,0.61,1.41,0.55,0.17,1.65,0.47,0.65,1.38,0.44,0.77,0.58,1.38,0.98,0.79,0.97,1.08  ],
  [0.52,0.48,1.46,1.3,0.43,0.61,1.37,0.58,0.18,1.67,0.42,0.6,1.38,0.45,0.77,0.74,1.43,1.01,0.79,0.95,0.92  ],
  [0.5,0.49,1.42,1.38,0.32,0.65,1.44,0.5,0.18,1.73,0.42,0.5,1.41,0.43,0.76,0.64,1.19,0.98,0.86,0.99,0.98  ],
  [0.53,0.55,1.48,1.44,0.35,0.58,1.47,0.57,0.18,1.68,0.41,0.54,1.46,0.43,0.88,0.61,1.32,0.99,0.66,0.99,1.04  ],
  [0.52,0.52,1.4,1.35,0.33,0.61,1.36,0.49,0.17,1.62,0.49,0.5,1.33,0.48,0.72,0.67,1.02,0.98,0.85,0.95,1  ],
  [0.57,0.53,1.3,1.32,0.29,0.59,1.35,0.45,0.17,1.66,0.43,0.55,1.35,0.42,0.76,0.64,1.28,1,0.78,1.04,0.93  ],
  [0.57,0.54,1.45,1.34,0.4,0.65,1.41,0.57,0.18,1.67,0.49,0.57,1.41,0.47,0.76,0.7,1.16,1,0.86,0.97,0.95  ],
  [0.54,0.53,1.42,1.36,0.31,0.6,1.38,0.46,0.18,1.64,0.44,0.54,1.37,0.43,0.77,0.67,1.23,0.99,0.78,0.96,0.98  ],
  [0.62,0.56,1.41,1.33,0.34,0.61,1.42,0.56,0.17,1.54,0.46,0.66,1.39,0.47,0.78,0.61,1.43,0.98,0.78,0.99,0.9  ],
  [0.55,0.53,1.45,1.34,0.32,0.54,1.37,0.47,0.17,1.7,0.44,0.45,1.37,0.42,0.83,0.68,1.02,1,0.65,0.94,0.96  ],
  [0.54,0.52,1.42,1.33,0.34,0.58,1.38,0.5,0.18,1.68,0.42,0.55,1.37,0.48,0.79,0.68,1.31,0.99,0.73,0.96,0.96  ],
  [0.52,0.55,1.44,1.38,0.35,0.55,1.43,0.46,0.2,1.62,0.42,0.4,1.31,0.44,0.76,0.76,0.95,0.92,0.72,0.91,1.06  ],
  [0.6,0.55,1.43,1.34,0.35,0.58,1.37,0.5,0.17,1.68,0.5,0.5,1.37,0.48,0.79,0.7,1,1,0.73,0.96,0.92  ],
  [0.52,0.57,1.35,1.32,0.34,0.57,1.36,0.47,0.17,1.62,0.41,0.51,1.35,0.43,0.78,0.72,1.24,0.99,0.73,1,1.1  ],
  [0.56,0.53,1.39,1.33,0.34,0.58,1.38,0.49,0.17,1.72,0.42,0.49,1.36,0.44,0.78,0.69,1.17,0.99,0.74,0.98,0.95  ],
  [0.55,0.54,1.37,1.33,0.29,0.57,1.37,0.44,0.17,1.61,0.43,0.53,1.35,0.43,0.78,0.66,1.23,0.99,0.73,0.99,0.98  ],
  [0.57,0.59,1.51,1.34,0.37,0.66,1.44,0.6,0.18,1.65,0.48,0.7,1.41,0.48,0.75,0.62,1.46,0.98,0.88,0.93,1.04  ],
  [0.56,0.55,1.36,1.32,0.35,0.59,1.36,0.57,0.17,1.6,0.52,0.5,1.35,0.46,0.76,0.61,0.96,0.99,0.78,0.99,0.98  ],
  [0.53,0.56,1.43,1.33,0.3,0.56,1.41,0.55,0.18,1.54,0.46,0.55,1.39,0.44,0.83,0.55,1.2,0.99,0.67,0.97,1.06  ],
  [0.54,0.53,1.41,1.34,0.36,0.59,1.4,0.54,0.17,1.66,0.44,0.59,1.39,0.43,0.8,0.67,1.34,0.99,0.74,0.99,0.98  ],
  [0.6,0.51,1.51,1.35,0.22,0.5,1.4,0.45,0.17,1.65,0.45,0.47,1.38,0.38,0.88,0.49,1.04,0.99,0.57,0.91,0.85  ],
  [0.58,0.56,1.45,1.37,0.37,0.65,1.45,0.59,0.18,1.72,0.47,0.59,1.43,0.46,0.78,0.63,1.26,0.99,0.83,0.99,0.97  ],
  [0.49,0.52,1.41,1.36,0.34,0.55,1.4,0.45,0.18,1.65,0.43,0.5,1.32,0.4,0.77,0.76,1.16,0.94,0.71,0.94,1.06  ],
  [0.52,0.55,1.43,1.33,0.35,0.63,1.41,0.58,0.18,1.58,0.48,0.6,1.4,0.43,0.77,0.6,1.25,0.99,0.82,0.98,1.06  ],
  [0.58,0.55,1.45,1.37,0.3,0.55,1.41,0.46,0.18,1.63,0.47,0.47,1.39,0.45,0.84,0.65,1,0.99,0.65,0.96,0.95  ],
  [0.52,0.54,1.37,1.34,0.3,0.56,1.35,0.46,0.17,1.67,0.43,0.5,1.36,0.41,0.8,0.65,1.16,1.01,0.7,0.99,1.04  ],
  [0.56,0.56,1.42,1.36,0.35,0.62,1.44,0.6,0.18,1.6,0.44,0.64,1.42,0.45,0.8,0.58,1.45,0.99,0.78,1,1  ],
  [0.57,0.59,1.48,1.39,0.34,0.65,1.46,0.6,0.18,1.6,0.5,0.64,1.45,0.43,0.8,0.57,1.28,0.99,0.81,0.98,1.04  ],
  [0.48,0.44,1.36,1.33,0.29,0.61,1.37,0.48,0.17,1.6,0.42,0.53,1.35,0.49,0.74,0.6,1.26,0.99,0.82,0.99,0.92  ],
  [0.52,0.5,1.39,1.36,0.29,0.59,1.38,0.45,0.17,1.66,0.42,0.51,1.36,0.42,0.77,0.64,1.21,0.99,0.77,0.98,0.96  ],
  [0.68,0.49,1.38,1.33,0.28,0.69,1.37,0.46,0.17,1.62,0.44,0.5,1.35,0.4,0.66,0.61,1.14,0.99,1.05,0.98,0.72  ],
  [0.54,0.54,1.4,1.32,0.31,0.6,1.37,0.47,0.17,1.61,0.39,0.49,1.35,0.41,0.75,0.66,1.26,0.99,0.8,0.96,1  ],
  [0.47,0.45,1.42,1.34,0.29,0.62,1.38,0.48,0.17,1.62,0.44,0.51,1.36,0.41,0.74,0.6,1.16,0.99,0.84,0.96,0.96  ],
  [0.52,0.51,1.39,1.36,0.34,0.54,1.39,0.52,0.17,1.64,0.41,0.5,1.37,0.4,0.83,0.65,1.22,0.99,0.65,0.99,0.98  ],
  [0.56,0.57,1.37,1.3,0.34,0.6,1.37,0.56,0.17,1.49,0.46,0.66,1.35,0.45,0.75,0.61,1.43,0.99,0.8,0.99,1.02  ],
  [0.53,0.52,1.47,1.43,0.34,0.62,1.48,0.54,0.19,1.62,0.42,0.53,1.45,0.42,0.83,0.63,1.26,0.98,0.75,0.99,0.98  ],
  [0.54,0.54,1.43,1.37,0.35,0.58,1.41,0.43,0.18,1.75,0.42,0.52,1.41,0.43,0.83,0.81,1.24,1,0.7,0.99,1  ],
  [0.58,0.57,1.43,1.31,0.31,0.62,1.4,0.58,0.17,1.51,0.46,0.64,1.38,0.46,0.76,0.53,1.39,0.99,0.82,0.97,0.98  ],
  [0.56,0.53,1.45,1.32,0.34,0.62,1.42,0.59,0.18,1.63,0.48,0.64,1.4,0.46,0.78,0.58,1.33,0.99,0.79,0.97,0.95  ],
  [0.52,0.52,1.42,1.32,0.29,0.56,1.36,0.42,0.17,1.65,0.43,0.5,1.35,0.4,0.79,0.69,1.16,0.99,0.71,0.95,1  ],
  [0.6,0.57,1.44,1.39,0.34,0.6,1.43,0.55,0.18,1.71,0.45,0.56,1.43,0.44,0.83,0.62,1.24,1,0.72,0.99,0.95  ],
  [0.6,0.56,1.45,1.38,0.39,0.61,1.45,0.48,0.18,1.61,0.46,0.68,1.43,0.48,0.82,0.81,1.48,0.99,0.74,0.99,0.93  ],
  [0.53,0.53,1.47,1.34,0.33,0.62,1.42,0.56,0.17,1.57,0.5,0.68,1.4,0.44,0.78,0.59,1.36,0.99,0.79,0.95,1  ],
  [0.51,0.51,1.33,1.25,0.31,0.55,1.27,0.4,0.16,1.54,0.39,0.47,1.27,0.42,0.72,0.78,1.21,1,0.76,0.95,1  ],
  [0.56,0.49,1.48,1.43,0.36,0.55,1.4,0.47,0.18,1.66,0.48,0.59,1.38,0.46,0.83,0.77,1.23,0.99,0.66,0.93,0.88  ],
  [0.57,0.54,1.51,1.36,0.41,0.63,1.47,0.53,0.18,1.77,0.52,0.62,1.45,0.51,0.82,0.773584906,1.1,0.986394558,0.768292683,0.960264901,0.947368421  ],
  [0.53,0.52,1.45,1.4,0.34,0.61,1.44,0.53,0.18,1.7,0.43,0.55,1.44,0.42,0.83,0.64,1.28,1,0.73,0.99,0.98  ],
  [0.54,0.56,1.45,1.4,0.32,0.6,0.45,0.47,0.19,1.69,0.45,0.52,1.39,0.44,0.79,0.68,1.16,3.09,0.76,0.96,1.04  ],
  [0.61,0.57,1.57,1.35,0.38,0.64,1.47,0.61,0.18,1.67,0.53,0.74,1.44,0.5,0.8,0.62,1.4,0.98,0.8,0.92,0.93  ],
  [0.55,0.55,1.43,1.35,0.33,0.55,1.38,0.47,0.18,1.69,0.47,0.43,1.38,0.46,0.83,0.7,0.91,1,0.66,0.97,1  ],
  [0.56,0.55,1.41,1.34,0.34,0.64,1.42,0.56,0.17,1.57,0.47,0.64,1.4,0.47,0.76,0.61,1.36,0.99,0.84,0.99,0.98  ],
  [0.54,0.5,1.4,1.31,0.31,0.57,1.35,0.44,0.17,1.6,0.42,0.51,1.35,0.43,0.78,0.7,1.21,1,0.73,0.96,0.93  ],
  [0.55,0.55,1.46,1.36,0.38,0.59,1.45,0.52,0.18,1.61,0.47,0.57,1.44,0.44,0.85,0.73,1.21,0.99,0.69,0.99,1  ],
  [0.54,0.54,1.36,1.36,0.34,0.6,1.31,0.47,0.17,1.64,0.43,0.48,1.31,0.48,0.71,0.72,1.12,1,0.85,0.96,1  ],
  [0.55,0.52,1.38,1.37,0.37,0.62,1.41,0.46,0.18,1.67,0.46,0.49,1.3,0.43,0.68,0.8,1.07,0.92,0.91,0.94,0.95  ],
  [0.54,0.55,1.36,1.3,0.32,0.59,1.35,0.43,0.17,1.58,0.42,0.56,1.35,0.44,0.76,0.74,1.33,1,0.78,0.99,1.02  ],
  [0.58,0.58,1.5,1.41,0.33,0.6,1.42,0.5,0.17,1.65,0.47,0.55,1.4,0.5,0.8,0.66,1.17,0.99,0.75,0.93,1  ],
  [0.55,0.56,1.41,1.34,0.34,0.64,1.43,0.57,0.17,1.6,0.48,0.62,1.4,0.46,0.76,0.6,1.29,0.98,0.84,0.99,1.02  ],
  [0.58,0.54,1.43,1.39,0.34,0.56,1.44,0.47,0.19,1.7,0.46,0.47,1.3,0.44,0.74,0.72,1.02,0.9,0.76,0.91,0.93  ],
  [0.54,0.55,1.42,1.34,0.28,0.55,1.39,0.46,0.17,1.66,0.44,0.57,1.37,0.42,0.82,0.61,1.3,0.99,0.67,0.96,1.02  ],
  [0.54,0.53,1.42,1.33,0.37,0.68,1.39,0.58,0.18,1.69,0.47,0.53,1.38,0.42,0.7,0.64,1.13,0.99,0.97,0.97,0.98  ]
]

  module.exports = todasMedidas