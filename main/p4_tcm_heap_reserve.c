/*
 * ESP32-P4 TCM is cache-independent memory, but ESP-IDF v5.5.4 flash/NVS
 * critical sections only accept task stacks in SOC_DRAM_* for cache-disabled
 * operations. Leaving free TCM in the generic INTERNAL|8BIT heap lets small
 * task stacks, including main_task, land at 0x3010xxxx and trip that assert.
 */
#include <stdint.h>

#include "heap_memory_layout.h"
#include "sdkconfig.h"
#include "soc/soc.h"

#if CONFIG_IDF_TARGET_ESP32P4
extern int _tcm_data_end;

SOC_RESERVE_MEMORY_REGION((intptr_t)&_tcm_data_end, SOC_TCM_HIGH, p4_free_tcm_no_heap);
#endif

void p4_tcm_heap_reserve_linker_hook(void)
{
}
