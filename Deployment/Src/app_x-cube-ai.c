
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
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

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\<user_name>\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "network.h"
#include "network_data.h"

/* USER CODE BEGIN includes */
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_NETWORK_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_NETWORK_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle network = AI_HANDLE_NULL;

AI_ALIGNED(32)
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

/* Array to store the data of the input tensor */
AI_ALIGNED(32)
static ai_float in_data[AI_NETWORK_IN_1_SIZE];
/* or static ai_u8 in_data[AI_NETWORK_IN_1_SIZE_BYTES]; */

/* c-array to store the data of the output tensor */
AI_ALIGNED(32)
static ai_float out_data[AI_NETWORK_OUT_1_SIZE];
/* static ai_u8 out_data[AI_NETWORK_OUT_1_SIZE_BYTES]; */

/* Array of pointer to manage the model's input/output tensors */
static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_network_create_and_init(&network, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_network_create_and_init");
    return -1;
  }

  ai_input = ai_network_inputs_get(network, NULL);
  ai_output = ai_network_outputs_get(network, NULL);

#if defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++) {
  data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++) {
    ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++) {
  data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++) {
  ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_network_run(network, ai_input, ai_output);
//  if (batch != 1) {
//    ai_log_err(ai_network_get_error(network),
//        "ai_network_run");
//    return -1;
//  }

  return 0;
}

/* USER CODE BEGIN 2 */
int acquire_and_process_data(ai_i8* data[])
{
  /* fill the inputs of the c-model
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
  return 0;
}

int post_process(ai_i8* data[])
{
  /* process the predictions
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
  return 0;
}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/
int aiInit(void) {
//  ai_error err;
  /* Create and initialize the c-model */
  const ai_handle acts[] = { activations };
	ai_network_create_and_init(&network, acts, NULL);
//  err = ai_network_create_and_init(&network, acts, NULL);
//  if (err.type != AI_ERROR_NONE)
//  {
//    return -1;
//  };
  /* Reteive pointers to the model's input/output tensors */
  ai_input = ai_network_inputs_get(network, NULL);
  ai_output = ai_network_outputs_get(network, NULL);

  return 0;
}

/*
 * Run inference
 */
int aiRun(const void *in_data, void *out_data) {
//  ai_i32 n_batch;
//  ai_error err;

  /* 1 - Update IO handlers with the data payload */
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].data = AI_HANDLE_PTR(out_data);

  /* 2 - Perform the inference */
	ai_network_run(network, &ai_input[0], &ai_output[0]);
//n_batch = ai_network_run(network, &ai_input[0], &ai_output[0]);
//  if (n_batch != 1) {
//      err = ai_network_get_error(network);
//      return err.code;
//  };

  return 0;
}
void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
    /* Activation/working buffer is allocated as a static memory chunk
     * (bss section) */
    AI_ALIGNED(4)
    static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

    aiInit();
    /* USER CODE END 5 */
}

//void MX_X_CUBE_AI_Process(void)
//{
//    /* USER CODE BEGIN 1 */
//  int nb_run = 20;
//    int res;

//    /* Example of definition of the buffers to store the tensor input/output */
//    /*  type is dependent of the expected format                             */
//    AI_ALIGNED(4)
//    static ai_i8 in_data[AI_NETWORK_IN_1_SIZE_BYTES];

//    AI_ALIGNED(4)
//    static ai_i8 out_data[AI_NETWORK_OUT_1_SIZE_BYTES];

//    /* Retrieve format/type of the first input tensor - index 0 */
//    const ai_buffer_format fmt_ = AI_BUFFER_FORMAT(&ai_input[0]);
//    const uint32_t type_ = AI_BUFFER_FMT_GET_TYPE(fmt_);

//    /* Prepare parameters for float to Qmn conversion */
//    const ai_i16 N_ = AI_BUFFER_FMT_GET_FBITS(fmt_);
//    const ai_float scale_ = (0x1U << N_);
//    const ai_i16 M_ =  AI_BUFFER_FMT_GET_BITS(fmt_)
//                       - AI_BUFFER_FMT_GET_SIGN(fmt_) - N_;
//    const ai_float max_ = (ai_float)(0x1U << M_);

//    /* Perform nb_rub inferences (batch = 1) */
//    while (--nb_run) {

//        /* ---------------------------------------- */
//        /* Data generation and Pre-Process          */
//        /* ---------------------------------------- */
//        /* - fill the input buffer with random data */
//        for (ai_size i=0;  i < AI_NETWORK_IN_1_SIZE; i++ ) {

//            /* Generate random data in the range [-1, 1] */
//            ai_float val = 2.0f * (ai_float)rand() / (ai_float)RAND_MAX - 1.0f;

//            /* Convert the data if necessary */
//            if (type_ == AI_BUFFER_FMT_TYPE_FLOAT) {
//                ((ai_float *)in_data)[i] = val;
//            } else { /* AI_BUFFER_FMT_TYPE_Q */
//                /* Scale the values in the range [-2^M, 2^M] */
//                val *= max_;
//                /* Convert float to Qmn format */
//                const ai_i32 tmp_ = AI_ROUND(val * scale_, ai_i32);
//                in_data[i] =  AI_CLAMP(tmp_, -128, 127, ai_i8);
//            }
//        }

//        /* Perform the inference */
//        res = aiRun(in_data, out_data);
//        if (res) {
//            return;
//        }

//        /* Post-Process - process the output buffer */
//        // ...
//    }
//    /* USER CODE END 1 */
//}
#ifdef __cplusplus
}
#endif
