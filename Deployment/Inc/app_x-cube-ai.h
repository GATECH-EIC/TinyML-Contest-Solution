
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __APP_AI_H
#define __APP_AI_H
#ifdef __cplusplus
extern "C" {
#endif
/**
  ******************************************************************************
  * @file    app_x-cube-ai.h
  * @author  X-CUBE-AI C code generator
  * @brief   AI entry function definitions
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* Includes ------------------------------------------------------------------*/
#include "ai_platform.h"

void MX_X_CUBE_AI_Init(void);
void MX_X_CUBE_AI_Process(void);
int aiInit(void);
int aiRun(const void *in_data, void *out_data);

/* USER CODE BEGIN includes */
/* USER CODE END includes */

/* Helper macro */
#define AI_MIN(x_, y_) \
  ( ((x_)<(y_)) ? (x_) : (y_) )

#define AI_MAX(x_, y_) \
  ( ((x_)>(y_)) ? (x_) : (y_) )

#define AI_CLAMP(x_, min_, max_, type_) \
  (type_) (AI_MIN(AI_MAX(x_, min_), max_))

#define AI_ROUND(v_, type_) \
  (type_) ( ((v_)<0) ? ((v_)-0.5f) : ((v_)+0.5f) )


#ifdef __cplusplus
}
#endif
#endif /*__STMicroelectronics_X-CUBE-AI_7_2_0_H */
