// SPDX-License-Identifier: GPL-2.0+
/*
 * (c) Copyright 2010-2014 Xilinx, Inc. All rights reserved.
 */

#include <asm/arch/ps7_init_gpl.h>

static unsigned long ps7_pll_init_data_3_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000110, 0x003FFFF0U, 0x000FA220U),
	EMIT_MASKWRITE(0xF8000100, 0x0007F000U, 0x00028000U),
	EMIT_MASKWRITE(0xF8000100, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000100, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000100, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000001U),
	EMIT_MASKWRITE(0xF8000100, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000120, 0x1F003F30U, 0x1F000200U),
	EMIT_MASKWRITE(0xF8000114, 0x003FFFF0U, 0x0012C220U),
	EMIT_MASKWRITE(0xF8000104, 0x0007F000U, 0x00020000U),
	EMIT_MASKWRITE(0xF8000104, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000104, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000104, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000002U),
	EMIT_MASKWRITE(0xF8000104, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000124, 0xFFF00003U, 0x0C200003U),
	EMIT_MASKWRITE(0xF8000118, 0x003FFFF0U, 0x001452C0U),
	EMIT_MASKWRITE(0xF8000108, 0x0007F000U, 0x0001E000U),
	EMIT_MASKWRITE(0xF8000108, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000108, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000108, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000004U),
	EMIT_MASKWRITE(0xF8000108, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_clock_init_data_3_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000128, 0x03F03F01U, 0x00700F01U),
	EMIT_MASKWRITE(0xF800013C, 0x00000011U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000144, 0x03F03F71U, 0x00100801U),
	EMIT_MASKWRITE(0xF800014C, 0x00003F31U, 0x00000501U),
	EMIT_MASKWRITE(0xF8000154, 0x00003F33U, 0x00001401U),
	EMIT_MASKWRITE(0xF8000158, 0x00003F33U, 0x00000601U),
	EMIT_MASKWRITE(0xF800015C, 0x03F03F33U, 0x00600702U),
	EMIT_MASKWRITE(0xF8000160, 0x007F007FU, 0x00000000U),
	EMIT_MASKWRITE(0xF8000168, 0x00003F31U, 0x00000401U),
	EMIT_MASKWRITE(0xF8000170, 0x03F03F30U, 0x00400500U),
	EMIT_MASKWRITE(0xF80001C4, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF800012C, 0x01FFCCCDU, 0x01DE408DU),
	EMIT_MASKWRITE(0xF8000304, 0x00000001U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_ddr_init_data_3_0[] = {
	EMIT_MASKWRITE(0xF8006000, 0x0001FFFFU, 0x00000080U),
	EMIT_MASKWRITE(0xF8006004, 0x0007FFFFU, 0x00001081U),
	EMIT_MASKWRITE(0xF8006008, 0x03FFFFFFU, 0x03C0780FU),
	EMIT_MASKWRITE(0xF800600C, 0x03FFFFFFU, 0x02001001U),
	EMIT_MASKWRITE(0xF8006010, 0x03FFFFFFU, 0x00014001U),
	EMIT_MASKWRITE(0xF8006014, 0x001FFFFFU, 0x0004159BU),
	EMIT_MASKWRITE(0xF8006018, 0xF7FFFFFFU, 0x44E438D2U),
	EMIT_MASKWRITE(0xF800601C, 0xFFFFFFFFU, 0x720238E5U),
	EMIT_MASKWRITE(0xF8006020, 0x7FDFFFFCU, 0x27087290U),
	EMIT_MASKWRITE(0xF8006024, 0x0FFFFFC3U, 0x00000000U),
	EMIT_MASKWRITE(0xF8006028, 0x00003FFFU, 0x00002007U),
	EMIT_MASKWRITE(0xF800602C, 0xFFFFFFFFU, 0x00000008U),
	EMIT_MASKWRITE(0xF8006030, 0xFFFFFFFFU, 0x00040930U),
	EMIT_MASKWRITE(0xF8006034, 0x13FF3FFFU, 0x000116D4U),
	EMIT_MASKWRITE(0xF8006038, 0x00000003U, 0x00000000U),
	EMIT_MASKWRITE(0xF800603C, 0x000FFFFFU, 0x00000777U),
	EMIT_MASKWRITE(0xF8006040, 0xFFFFFFFFU, 0xFFF00000U),
	EMIT_MASKWRITE(0xF8006044, 0x0FFFFFFFU, 0x0F666666U),
	EMIT_MASKWRITE(0xF8006048, 0x0003F03FU, 0x0003C008U),
	EMIT_MASKWRITE(0xF8006050, 0xFF0F8FFFU, 0x77010800U),
	EMIT_MASKWRITE(0xF8006058, 0x00010000U, 0x00000000U),
	EMIT_MASKWRITE(0xF800605C, 0x0000FFFFU, 0x00005003U),
	EMIT_MASKWRITE(0xF8006060, 0x000017FFU, 0x0000003EU),
	EMIT_MASKWRITE(0xF8006064, 0x00021FE0U, 0x00020000U),
	EMIT_MASKWRITE(0xF8006068, 0x03FFFFFFU, 0x00284141U),
	EMIT_MASKWRITE(0xF800606C, 0x0000FFFFU, 0x00001610U),
	EMIT_MASKWRITE(0xF8006078, 0x03FFFFFFU, 0x00466111U),
	EMIT_MASKWRITE(0xF800607C, 0x000FFFFFU, 0x00032222U),
	EMIT_MASKWRITE(0xF80060A4, 0xFFFFFFFFU, 0x10200802U),
	EMIT_MASKWRITE(0xF80060A8, 0x0FFFFFFFU, 0x0690CB73U),
	EMIT_MASKWRITE(0xF80060AC, 0x000001FFU, 0x000001FEU),
	EMIT_MASKWRITE(0xF80060B0, 0x1FFFFFFFU, 0x1CFFFFFFU),
	EMIT_MASKWRITE(0xF80060B4, 0x00000200U, 0x00000200U),
	EMIT_MASKWRITE(0xF80060B8, 0x01FFFFFFU, 0x00200066U),
	EMIT_MASKWRITE(0xF80060C4, 0x00000003U, 0x00000003U),
	EMIT_MASKWRITE(0xF80060C4, 0x00000003U, 0x00000000U),
	EMIT_MASKWRITE(0xF80060C8, 0x000000FFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80060DC, 0x00000001U, 0x00000000U),
	EMIT_MASKWRITE(0xF80060F0, 0x0000FFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80060F4, 0x0000000FU, 0x00000008U),
	EMIT_MASKWRITE(0xF8006114, 0x000000FFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8006118, 0x7FFFFFCFU, 0x40000001U),
	EMIT_MASKWRITE(0xF800611C, 0x7FFFFFCFU, 0x40000001U),
	EMIT_MASKWRITE(0xF8006120, 0x7FFFFFCFU, 0x40000001U),
	EMIT_MASKWRITE(0xF8006124, 0x7FFFFFCFU, 0x40000001U),
	EMIT_MASKWRITE(0xF800612C, 0x000FFFFFU, 0x00039C1BU),
	EMIT_MASKWRITE(0xF8006130, 0x000FFFFFU, 0x00037C35U),
	EMIT_MASKWRITE(0xF8006134, 0x000FFFFFU, 0x0003942FU),
	EMIT_MASKWRITE(0xF8006138, 0x000FFFFFU, 0x00038C1FU),
	EMIT_MASKWRITE(0xF8006140, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006144, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006148, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF800614C, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006154, 0x000FFFFFU, 0x0000009BU),
	EMIT_MASKWRITE(0xF8006158, 0x000FFFFFU, 0x000000B5U),
	EMIT_MASKWRITE(0xF800615C, 0x000FFFFFU, 0x000000AFU),
	EMIT_MASKWRITE(0xF8006160, 0x000FFFFFU, 0x0000009FU),
	EMIT_MASKWRITE(0xF8006168, 0x001FFFFFU, 0x0000013CU),
	EMIT_MASKWRITE(0xF800616C, 0x001FFFFFU, 0x00000134U),
	EMIT_MASKWRITE(0xF8006170, 0x001FFFFFU, 0x0000013AU),
	EMIT_MASKWRITE(0xF8006174, 0x001FFFFFU, 0x00000138U),
	EMIT_MASKWRITE(0xF800617C, 0x000FFFFFU, 0x000000DBU),
	EMIT_MASKWRITE(0xF8006180, 0x000FFFFFU, 0x000000F5U),
	EMIT_MASKWRITE(0xF8006184, 0x000FFFFFU, 0x000000EFU),
	EMIT_MASKWRITE(0xF8006188, 0x000FFFFFU, 0x000000DFU),
	EMIT_MASKWRITE(0xF8006190, 0x6FFFFEFEU, 0x00040080U),
	EMIT_MASKWRITE(0xF8006194, 0x000FFFFFU, 0x0001FC82U),
	EMIT_MASKWRITE(0xF8006204, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8006208, 0x000703FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF800620C, 0x000703FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006210, 0x000703FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006214, 0x000703FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006218, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF800621C, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006220, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006224, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF80062A8, 0x00000FF5U, 0x00000000U),
	EMIT_MASKWRITE(0xF80062AC, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80062B0, 0x003FFFFFU, 0x00005125U),
	EMIT_MASKWRITE(0xF80062B4, 0x0003FFFFU, 0x000012A8U),
	EMIT_MASKPOLL(0xF8000B74, 0x00002000U),
	EMIT_MASKWRITE(0xF8006000, 0x0001FFFFU, 0x00000081U),
	EMIT_MASKPOLL(0xF8006054, 0x00000007U),
	EMIT_EXIT(),
};

static unsigned long ps7_mio_init_data_3_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000B40, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B44, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B48, 0x00000FFFU, 0x00000672U),
	EMIT_MASKWRITE(0xF8000B4C, 0x00000FFFU, 0x00000672U),
	EMIT_MASKWRITE(0xF8000B50, 0x00000FFFU, 0x00000674U),
	EMIT_MASKWRITE(0xF8000B54, 0x00000FFFU, 0x00000674U),
	EMIT_MASKWRITE(0xF8000B58, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B5C, 0xFFFFFFFFU, 0x0018C61CU),
	EMIT_MASKWRITE(0xF8000B60, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B64, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B68, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B6C, 0x00007FFFU, 0x00000209U),
	EMIT_MASKWRITE(0xF8000B70, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000B70, 0x00000021U, 0x00000020U),
	EMIT_MASKWRITE(0xF8000B70, 0x07FEFFFFU, 0x00000823U),
	EMIT_MASKWRITE(0xF8000700, 0x00003FFFU, 0x00001602U),
	EMIT_MASKWRITE(0xF8000704, 0x00003FFFU, 0x00001602U),
	EMIT_MASKWRITE(0xF8000708, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF800070C, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000710, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000714, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000718, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000720, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000738, 0x00003FFFU, 0x000006E1U),
	EMIT_MASKWRITE(0xF800073C, 0x00003FFFU, 0x000006E0U),
	EMIT_MASKWRITE(0xF8000740, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000744, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000748, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF800074C, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000750, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000754, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000770, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000774, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000778, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF800077C, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000780, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000784, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000788, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF800078C, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000790, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000794, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000798, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF800079C, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF80007A0, 0x00003FFFU, 0x00000720U),
	EMIT_MASKWRITE(0xF80007A4, 0x00003FFFU, 0x00000721U),
	EMIT_MASKWRITE(0xF80007A8, 0x00003FFFU, 0x000007C0U),
	EMIT_MASKWRITE(0xF80007AC, 0x00003FFFU, 0x000007C1U),
	EMIT_MASKWRITE(0xF80007B0, 0x00003FFFU, 0x00000740U),
	EMIT_MASKWRITE(0xF80007B4, 0x00003FFFU, 0x00000740U),
	EMIT_MASKWRITE(0xF80007B8, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007BC, 0x00003FFFU, 0x00000660U),
	EMIT_MASKWRITE(0xF80007C0, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007C4, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007C8, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007CC, 0x00003FFFU, 0x00000660U),
	EMIT_MASKWRITE(0xF80007D0, 0x00003FFFU, 0x000006A0U),
	EMIT_MASKWRITE(0xF80007D4, 0x00003FFFU, 0x000006A0U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_peripherals_init_data_3_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000B48, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B4C, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B50, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B54, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_MASKWRITE(0xE0000034, 0x000000FFU, 0x00000006U),
	EMIT_MASKWRITE(0xE0000018, 0x0000FFFFU, 0x0000003EU),
	EMIT_MASKWRITE(0xE0000000, 0x000001FFU, 0x00000017U),
	EMIT_MASKWRITE(0xE0000004, 0x000003FFU, 0x00000020U),
	EMIT_MASKWRITE(0xE000D000, 0x00080000U, 0x00080000U),
	EMIT_MASKWRITE(0xF8007000, 0x20000000U, 0x00000000U),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_EXIT(),
};

static unsigned long ps7_post_config_3_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000900, 0x0000000FU, 0x0000000FU),
	EMIT_MASKWRITE(0xF8000240, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_pll_init_data_2_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000110, 0x003FFFF0U, 0x000FA220U),
	EMIT_MASKWRITE(0xF8000100, 0x0007F000U, 0x00028000U),
	EMIT_MASKWRITE(0xF8000100, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000100, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000100, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000001U),
	EMIT_MASKWRITE(0xF8000100, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000120, 0x1F003F30U, 0x1F000200U),
	EMIT_MASKWRITE(0xF8000114, 0x003FFFF0U, 0x0012C220U),
	EMIT_MASKWRITE(0xF8000104, 0x0007F000U, 0x00020000U),
	EMIT_MASKWRITE(0xF8000104, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000104, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000104, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000002U),
	EMIT_MASKWRITE(0xF8000104, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000124, 0xFFF00003U, 0x0C200003U),
	EMIT_MASKWRITE(0xF8000118, 0x003FFFF0U, 0x001452C0U),
	EMIT_MASKWRITE(0xF8000108, 0x0007F000U, 0x0001E000U),
	EMIT_MASKWRITE(0xF8000108, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000108, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000108, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000004U),
	EMIT_MASKWRITE(0xF8000108, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_clock_init_data_2_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000128, 0x03F03F01U, 0x00700F01U),
	EMIT_MASKWRITE(0xF800013C, 0x00000011U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000144, 0x03F03F71U, 0x00100801U),
	EMIT_MASKWRITE(0xF800014C, 0x00003F31U, 0x00000501U),
	EMIT_MASKWRITE(0xF8000154, 0x00003F33U, 0x00001401U),
	EMIT_MASKWRITE(0xF8000158, 0x00003F33U, 0x00000601U),
	EMIT_MASKWRITE(0xF800015C, 0x03F03F33U, 0x00600702U),
	EMIT_MASKWRITE(0xF8000160, 0x007F007FU, 0x00000000U),
	EMIT_MASKWRITE(0xF8000168, 0x00003F31U, 0x00000401U),
	EMIT_MASKWRITE(0xF8000170, 0x03F03F30U, 0x00400500U),
	EMIT_MASKWRITE(0xF80001C4, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF800012C, 0x01FFCCCDU, 0x01DE408DU),
	EMIT_MASKWRITE(0xF8000304, 0x00000001U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_ddr_init_data_2_0[] = {
	EMIT_MASKWRITE(0xF8006000, 0x0001FFFFU, 0x00000080U),
	EMIT_MASKWRITE(0xF8006004, 0x1FFFFFFFU, 0x00081081U),
	EMIT_MASKWRITE(0xF8006008, 0x03FFFFFFU, 0x03C0780FU),
	EMIT_MASKWRITE(0xF800600C, 0x03FFFFFFU, 0x02001001U),
	EMIT_MASKWRITE(0xF8006010, 0x03FFFFFFU, 0x00014001U),
	EMIT_MASKWRITE(0xF8006014, 0x001FFFFFU, 0x0004159BU),
	EMIT_MASKWRITE(0xF8006018, 0xF7FFFFFFU, 0x44E438D2U),
	EMIT_MASKWRITE(0xF800601C, 0xFFFFFFFFU, 0x720238E5U),
	EMIT_MASKWRITE(0xF8006020, 0xFFFFFFFCU, 0x27287290U),
	EMIT_MASKWRITE(0xF8006024, 0x0FFFFFFFU, 0x0000003CU),
	EMIT_MASKWRITE(0xF8006028, 0x00003FFFU, 0x00002007U),
	EMIT_MASKWRITE(0xF800602C, 0xFFFFFFFFU, 0x00000008U),
	EMIT_MASKWRITE(0xF8006030, 0xFFFFFFFFU, 0x00040930U),
	EMIT_MASKWRITE(0xF8006034, 0x13FF3FFFU, 0x000116D4U),
	EMIT_MASKWRITE(0xF8006038, 0x00001FC3U, 0x00000000U),
	EMIT_MASKWRITE(0xF800603C, 0x000FFFFFU, 0x00000777U),
	EMIT_MASKWRITE(0xF8006040, 0xFFFFFFFFU, 0xFFF00000U),
	EMIT_MASKWRITE(0xF8006044, 0x0FFFFFFFU, 0x0F666666U),
	EMIT_MASKWRITE(0xF8006048, 0x3FFFFFFFU, 0x0003C248U),
	EMIT_MASKWRITE(0xF8006050, 0xFF0F8FFFU, 0x77010800U),
	EMIT_MASKWRITE(0xF8006058, 0x0001FFFFU, 0x00000101U),
	EMIT_MASKWRITE(0xF800605C, 0x0000FFFFU, 0x00005003U),
	EMIT_MASKWRITE(0xF8006060, 0x000017FFU, 0x0000003EU),
	EMIT_MASKWRITE(0xF8006064, 0x00021FE0U, 0x00020000U),
	EMIT_MASKWRITE(0xF8006068, 0x03FFFFFFU, 0x00284141U),
	EMIT_MASKWRITE(0xF800606C, 0x0000FFFFU, 0x00001610U),
	EMIT_MASKWRITE(0xF8006078, 0x03FFFFFFU, 0x00466111U),
	EMIT_MASKWRITE(0xF800607C, 0x000FFFFFU, 0x00032222U),
	EMIT_MASKWRITE(0xF80060A0, 0x00FFFFFFU, 0x00008000U),
	EMIT_MASKWRITE(0xF80060A4, 0xFFFFFFFFU, 0x10200802U),
	EMIT_MASKWRITE(0xF80060A8, 0x0FFFFFFFU, 0x0690CB73U),
	EMIT_MASKWRITE(0xF80060AC, 0x000001FFU, 0x000001FEU),
	EMIT_MASKWRITE(0xF80060B0, 0x1FFFFFFFU, 0x1CFFFFFFU),
	EMIT_MASKWRITE(0xF80060B4, 0x000007FFU, 0x00000200U),
	EMIT_MASKWRITE(0xF80060B8, 0x01FFFFFFU, 0x00200066U),
	EMIT_MASKWRITE(0xF80060C4, 0x00000003U, 0x00000003U),
	EMIT_MASKWRITE(0xF80060C4, 0x00000003U, 0x00000000U),
	EMIT_MASKWRITE(0xF80060C8, 0x000000FFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80060DC, 0x00000001U, 0x00000000U),
	EMIT_MASKWRITE(0xF80060F0, 0x0000FFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80060F4, 0x0000000FU, 0x00000008U),
	EMIT_MASKWRITE(0xF8006114, 0x000000FFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8006118, 0x7FFFFFFFU, 0x40000001U),
	EMIT_MASKWRITE(0xF800611C, 0x7FFFFFFFU, 0x40000001U),
	EMIT_MASKWRITE(0xF8006120, 0x7FFFFFFFU, 0x40000001U),
	EMIT_MASKWRITE(0xF8006124, 0x7FFFFFFFU, 0x40000001U),
	EMIT_MASKWRITE(0xF800612C, 0x000FFFFFU, 0x00039C1BU),
	EMIT_MASKWRITE(0xF8006130, 0x000FFFFFU, 0x00037C35U),
	EMIT_MASKWRITE(0xF8006134, 0x000FFFFFU, 0x0003942FU),
	EMIT_MASKWRITE(0xF8006138, 0x000FFFFFU, 0x00038C1FU),
	EMIT_MASKWRITE(0xF8006140, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006144, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006148, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF800614C, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006154, 0x000FFFFFU, 0x0000009BU),
	EMIT_MASKWRITE(0xF8006158, 0x000FFFFFU, 0x000000B5U),
	EMIT_MASKWRITE(0xF800615C, 0x000FFFFFU, 0x000000AFU),
	EMIT_MASKWRITE(0xF8006160, 0x000FFFFFU, 0x0000009FU),
	EMIT_MASKWRITE(0xF8006168, 0x001FFFFFU, 0x0000013CU),
	EMIT_MASKWRITE(0xF800616C, 0x001FFFFFU, 0x00000134U),
	EMIT_MASKWRITE(0xF8006170, 0x001FFFFFU, 0x0000013AU),
	EMIT_MASKWRITE(0xF8006174, 0x001FFFFFU, 0x00000138U),
	EMIT_MASKWRITE(0xF800617C, 0x000FFFFFU, 0x000000DBU),
	EMIT_MASKWRITE(0xF8006180, 0x000FFFFFU, 0x000000F5U),
	EMIT_MASKWRITE(0xF8006184, 0x000FFFFFU, 0x000000EFU),
	EMIT_MASKWRITE(0xF8006188, 0x000FFFFFU, 0x000000DFU),
	EMIT_MASKWRITE(0xF8006190, 0xFFFFFFFFU, 0x10040080U),
	EMIT_MASKWRITE(0xF8006194, 0x000FFFFFU, 0x0001FC82U),
	EMIT_MASKWRITE(0xF8006204, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8006208, 0x000F03FFU, 0x000803FFU),
	EMIT_MASKWRITE(0xF800620C, 0x000F03FFU, 0x000803FFU),
	EMIT_MASKWRITE(0xF8006210, 0x000F03FFU, 0x000803FFU),
	EMIT_MASKWRITE(0xF8006214, 0x000F03FFU, 0x000803FFU),
	EMIT_MASKWRITE(0xF8006218, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF800621C, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006220, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006224, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF80062A8, 0x00000FF7U, 0x00000000U),
	EMIT_MASKWRITE(0xF80062AC, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80062B0, 0x003FFFFFU, 0x00005125U),
	EMIT_MASKWRITE(0xF80062B4, 0x0003FFFFU, 0x000012A8U),
	EMIT_MASKPOLL(0xF8000B74, 0x00002000U),
	EMIT_MASKWRITE(0xF8006000, 0x0001FFFFU, 0x00000081U),
	EMIT_MASKPOLL(0xF8006054, 0x00000007U),
	EMIT_EXIT(),
};

static unsigned long ps7_mio_init_data_2_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000B40, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B44, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B48, 0x00000FFFU, 0x00000672U),
	EMIT_MASKWRITE(0xF8000B4C, 0x00000FFFU, 0x00000672U),
	EMIT_MASKWRITE(0xF8000B50, 0x00000FFFU, 0x00000674U),
	EMIT_MASKWRITE(0xF8000B54, 0x00000FFFU, 0x00000674U),
	EMIT_MASKWRITE(0xF8000B58, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B5C, 0xFFFFFFFFU, 0x0018C61CU),
	EMIT_MASKWRITE(0xF8000B60, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B64, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B68, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B6C, 0x00007FFFU, 0x00000209U),
	EMIT_MASKWRITE(0xF8000B70, 0x00000021U, 0x00000021U),
	EMIT_MASKWRITE(0xF8000B70, 0x00000021U, 0x00000020U),
	EMIT_MASKWRITE(0xF8000B70, 0x07FFFFFFU, 0x00000823U),
	EMIT_MASKWRITE(0xF8000700, 0x00003FFFU, 0x00001602U),
	EMIT_MASKWRITE(0xF8000704, 0x00003FFFU, 0x00001602U),
	EMIT_MASKWRITE(0xF8000708, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF800070C, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000710, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000714, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000718, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000720, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000738, 0x00003FFFU, 0x000006E1U),
	EMIT_MASKWRITE(0xF800073C, 0x00003FFFU, 0x000006E0U),
	EMIT_MASKWRITE(0xF8000740, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000744, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000748, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF800074C, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000750, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000754, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000770, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000774, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000778, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF800077C, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000780, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000784, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000788, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF800078C, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000790, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000794, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000798, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF800079C, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF80007A0, 0x00003FFFU, 0x00000720U),
	EMIT_MASKWRITE(0xF80007A4, 0x00003FFFU, 0x00000721U),
	EMIT_MASKWRITE(0xF80007A8, 0x00003FFFU, 0x000007C0U),
	EMIT_MASKWRITE(0xF80007AC, 0x00003FFFU, 0x000007C1U),
	EMIT_MASKWRITE(0xF80007B0, 0x00003FFFU, 0x00000740U),
	EMIT_MASKWRITE(0xF80007B4, 0x00003FFFU, 0x00000740U),
	EMIT_MASKWRITE(0xF80007B8, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007BC, 0x00003FFFU, 0x00000660U),
	EMIT_MASKWRITE(0xF80007C0, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007C4, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007C8, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007CC, 0x00003FFFU, 0x00000660U),
	EMIT_MASKWRITE(0xF80007D0, 0x00003FFFU, 0x000006A0U),
	EMIT_MASKWRITE(0xF80007D4, 0x00003FFFU, 0x000006A0U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_peripherals_init_data_2_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000B48, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B4C, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B50, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B54, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_MASKWRITE(0xE0000034, 0x000000FFU, 0x00000006U),
	EMIT_MASKWRITE(0xE0000018, 0x0000FFFFU, 0x0000003EU),
	EMIT_MASKWRITE(0xE0000000, 0x000001FFU, 0x00000017U),
	EMIT_MASKWRITE(0xE0000004, 0x00000FFFU, 0x00000020U),
	EMIT_MASKWRITE(0xE000D000, 0x00080000U, 0x00080000U),
	EMIT_MASKWRITE(0xF8007000, 0x20000000U, 0x00000000U),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_EXIT(),
};

static unsigned long ps7_post_config_2_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000900, 0x0000000FU, 0x0000000FU),
	EMIT_MASKWRITE(0xF8000240, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_pll_init_data_1_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000110, 0x003FFFF0U, 0x000FA220U),
	EMIT_MASKWRITE(0xF8000100, 0x0007F000U, 0x00028000U),
	EMIT_MASKWRITE(0xF8000100, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000100, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000100, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000001U),
	EMIT_MASKWRITE(0xF8000100, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000120, 0x1F003F30U, 0x1F000200U),
	EMIT_MASKWRITE(0xF8000114, 0x003FFFF0U, 0x0012C220U),
	EMIT_MASKWRITE(0xF8000104, 0x0007F000U, 0x00020000U),
	EMIT_MASKWRITE(0xF8000104, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000104, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000104, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000002U),
	EMIT_MASKWRITE(0xF8000104, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000124, 0xFFF00003U, 0x0C200003U),
	EMIT_MASKWRITE(0xF8000118, 0x003FFFF0U, 0x001452C0U),
	EMIT_MASKWRITE(0xF8000108, 0x0007F000U, 0x0001E000U),
	EMIT_MASKWRITE(0xF8000108, 0x00000010U, 0x00000010U),
	EMIT_MASKWRITE(0xF8000108, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000108, 0x00000001U, 0x00000000U),
	EMIT_MASKPOLL(0xF800010C, 0x00000004U),
	EMIT_MASKWRITE(0xF8000108, 0x00000010U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_clock_init_data_1_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000128, 0x03F03F01U, 0x00700F01U),
	EMIT_MASKWRITE(0xF800013C, 0x00000011U, 0x00000001U),
	EMIT_MASKWRITE(0xF8000144, 0x03F03F71U, 0x00100801U),
	EMIT_MASKWRITE(0xF800014C, 0x00003F31U, 0x00000501U),
	EMIT_MASKWRITE(0xF8000154, 0x00003F33U, 0x00001401U),
	EMIT_MASKWRITE(0xF8000158, 0x00003F33U, 0x00000601U),
	EMIT_MASKWRITE(0xF800015C, 0x03F03F33U, 0x00600702U),
	EMIT_MASKWRITE(0xF8000160, 0x007F007FU, 0x00000000U),
	EMIT_MASKWRITE(0xF8000168, 0x00003F31U, 0x00000401U),
	EMIT_MASKWRITE(0xF8000170, 0x03F03F30U, 0x00400500U),
	EMIT_MASKWRITE(0xF80001C4, 0x00000001U, 0x00000001U),
	EMIT_MASKWRITE(0xF800012C, 0x01FFCCCDU, 0x01DE408DU),
	EMIT_MASKWRITE(0xF8000304, 0x00000001U, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_ddr_init_data_1_0[] = {
	EMIT_MASKWRITE(0xF8006000, 0x0001FFFFU, 0x00000080U),
	EMIT_MASKWRITE(0xF8006004, 0x1FFFFFFFU, 0x00081081U),
	EMIT_MASKWRITE(0xF8006008, 0x03FFFFFFU, 0x03C0780FU),
	EMIT_MASKWRITE(0xF800600C, 0x03FFFFFFU, 0x02001001U),
	EMIT_MASKWRITE(0xF8006010, 0x03FFFFFFU, 0x00014001U),
	EMIT_MASKWRITE(0xF8006014, 0x001FFFFFU, 0x0004159BU),
	EMIT_MASKWRITE(0xF8006018, 0xF7FFFFFFU, 0x44E438D2U),
	EMIT_MASKWRITE(0xF800601C, 0xFFFFFFFFU, 0x720238E5U),
	EMIT_MASKWRITE(0xF8006020, 0xFFFFFFFCU, 0x27287290U),
	EMIT_MASKWRITE(0xF8006024, 0x0FFFFFFFU, 0x0000003CU),
	EMIT_MASKWRITE(0xF8006028, 0x00003FFFU, 0x00002007U),
	EMIT_MASKWRITE(0xF800602C, 0xFFFFFFFFU, 0x00000008U),
	EMIT_MASKWRITE(0xF8006030, 0xFFFFFFFFU, 0x00040930U),
	EMIT_MASKWRITE(0xF8006034, 0x13FF3FFFU, 0x000116D4U),
	EMIT_MASKWRITE(0xF8006038, 0x00001FC3U, 0x00000000U),
	EMIT_MASKWRITE(0xF800603C, 0x000FFFFFU, 0x00000777U),
	EMIT_MASKWRITE(0xF8006040, 0xFFFFFFFFU, 0xFFF00000U),
	EMIT_MASKWRITE(0xF8006044, 0x0FFFFFFFU, 0x0F666666U),
	EMIT_MASKWRITE(0xF8006048, 0x3FFFFFFFU, 0x0003C248U),
	EMIT_MASKWRITE(0xF8006050, 0xFF0F8FFFU, 0x77010800U),
	EMIT_MASKWRITE(0xF8006058, 0x0001FFFFU, 0x00000101U),
	EMIT_MASKWRITE(0xF800605C, 0x0000FFFFU, 0x00005003U),
	EMIT_MASKWRITE(0xF8006060, 0x000017FFU, 0x0000003EU),
	EMIT_MASKWRITE(0xF8006064, 0x00021FE0U, 0x00020000U),
	EMIT_MASKWRITE(0xF8006068, 0x03FFFFFFU, 0x00284141U),
	EMIT_MASKWRITE(0xF800606C, 0x0000FFFFU, 0x00001610U),
	EMIT_MASKWRITE(0xF80060A0, 0x00FFFFFFU, 0x00008000U),
	EMIT_MASKWRITE(0xF80060A4, 0xFFFFFFFFU, 0x10200802U),
	EMIT_MASKWRITE(0xF80060A8, 0x0FFFFFFFU, 0x0690CB73U),
	EMIT_MASKWRITE(0xF80060AC, 0x000001FFU, 0x000001FEU),
	EMIT_MASKWRITE(0xF80060B0, 0x1FFFFFFFU, 0x1CFFFFFFU),
	EMIT_MASKWRITE(0xF80060B4, 0x000007FFU, 0x00000200U),
	EMIT_MASKWRITE(0xF80060B8, 0x01FFFFFFU, 0x00200066U),
	EMIT_MASKWRITE(0xF80060C4, 0x00000003U, 0x00000003U),
	EMIT_MASKWRITE(0xF80060C4, 0x00000003U, 0x00000000U),
	EMIT_MASKWRITE(0xF80060C8, 0x000000FFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80060DC, 0x00000001U, 0x00000000U),
	EMIT_MASKWRITE(0xF80060F0, 0x0000FFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80060F4, 0x0000000FU, 0x00000008U),
	EMIT_MASKWRITE(0xF8006114, 0x000000FFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8006118, 0x7FFFFFFFU, 0x40000001U),
	EMIT_MASKWRITE(0xF800611C, 0x7FFFFFFFU, 0x40000001U),
	EMIT_MASKWRITE(0xF8006120, 0x7FFFFFFFU, 0x40000001U),
	EMIT_MASKWRITE(0xF8006124, 0x7FFFFFFFU, 0x40000001U),
	EMIT_MASKWRITE(0xF800612C, 0x000FFFFFU, 0x00039C1BU),
	EMIT_MASKWRITE(0xF8006130, 0x000FFFFFU, 0x00037C35U),
	EMIT_MASKWRITE(0xF8006134, 0x000FFFFFU, 0x0003942FU),
	EMIT_MASKWRITE(0xF8006138, 0x000FFFFFU, 0x00038C1FU),
	EMIT_MASKWRITE(0xF8006140, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006144, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006148, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF800614C, 0x000FFFFFU, 0x00000035U),
	EMIT_MASKWRITE(0xF8006154, 0x000FFFFFU, 0x0000009BU),
	EMIT_MASKWRITE(0xF8006158, 0x000FFFFFU, 0x000000B5U),
	EMIT_MASKWRITE(0xF800615C, 0x000FFFFFU, 0x000000AFU),
	EMIT_MASKWRITE(0xF8006160, 0x000FFFFFU, 0x0000009FU),
	EMIT_MASKWRITE(0xF8006168, 0x001FFFFFU, 0x0000013CU),
	EMIT_MASKWRITE(0xF800616C, 0x001FFFFFU, 0x00000134U),
	EMIT_MASKWRITE(0xF8006170, 0x001FFFFFU, 0x0000013AU),
	EMIT_MASKWRITE(0xF8006174, 0x001FFFFFU, 0x00000138U),
	EMIT_MASKWRITE(0xF800617C, 0x000FFFFFU, 0x000000DBU),
	EMIT_MASKWRITE(0xF8006180, 0x000FFFFFU, 0x000000F5U),
	EMIT_MASKWRITE(0xF8006184, 0x000FFFFFU, 0x000000EFU),
	EMIT_MASKWRITE(0xF8006188, 0x000FFFFFU, 0x000000DFU),
	EMIT_MASKWRITE(0xF8006190, 0xFFFFFFFFU, 0x10040080U),
	EMIT_MASKWRITE(0xF8006194, 0x000FFFFFU, 0x0001FC82U),
	EMIT_MASKWRITE(0xF8006204, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8006208, 0x000F03FFU, 0x000803FFU),
	EMIT_MASKWRITE(0xF800620C, 0x000F03FFU, 0x000803FFU),
	EMIT_MASKWRITE(0xF8006210, 0x000F03FFU, 0x000803FFU),
	EMIT_MASKWRITE(0xF8006214, 0x000F03FFU, 0x000803FFU),
	EMIT_MASKWRITE(0xF8006218, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF800621C, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006220, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF8006224, 0x000F03FFU, 0x000003FFU),
	EMIT_MASKWRITE(0xF80062A8, 0x00000FF7U, 0x00000000U),
	EMIT_MASKWRITE(0xF80062AC, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF80062B0, 0x003FFFFFU, 0x00005125U),
	EMIT_MASKWRITE(0xF80062B4, 0x0003FFFFU, 0x000012A8U),
	EMIT_MASKPOLL(0xF8000B74, 0x00002000U),
	EMIT_MASKWRITE(0xF8006000, 0x0001FFFFU, 0x00000081U),
	EMIT_MASKPOLL(0xF8006054, 0x00000007U),
	EMIT_EXIT(),
};

static unsigned long ps7_mio_init_data_1_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000B40, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B44, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B48, 0x00000FFFU, 0x00000672U),
	EMIT_MASKWRITE(0xF8000B4C, 0x00000FFFU, 0x00000672U),
	EMIT_MASKWRITE(0xF8000B50, 0x00000FFFU, 0x00000674U),
	EMIT_MASKWRITE(0xF8000B54, 0x00000FFFU, 0x00000674U),
	EMIT_MASKWRITE(0xF8000B58, 0x00000FFFU, 0x00000600U),
	EMIT_MASKWRITE(0xF8000B5C, 0xFFFFFFFFU, 0x0018C61CU),
	EMIT_MASKWRITE(0xF8000B60, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B64, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B68, 0xFFFFFFFFU, 0x00F9861CU),
	EMIT_MASKWRITE(0xF8000B6C, 0x000073FFU, 0x00000209U),
	EMIT_MASKWRITE(0xF8000B70, 0x00000021U, 0x00000021U),
	EMIT_MASKWRITE(0xF8000B70, 0x00000021U, 0x00000020U),
	EMIT_MASKWRITE(0xF8000B70, 0x07FFFFFFU, 0x00000823U),
	EMIT_MASKWRITE(0xF8000700, 0x00003FFFU, 0x00001602U),
	EMIT_MASKWRITE(0xF8000704, 0x00003FFFU, 0x00001602U),
	EMIT_MASKWRITE(0xF8000708, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF800070C, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000710, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000714, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000718, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000720, 0x00003FFFU, 0x00000602U),
	EMIT_MASKWRITE(0xF8000738, 0x00003FFFU, 0x000006E1U),
	EMIT_MASKWRITE(0xF800073C, 0x00003FFFU, 0x000006E0U),
	EMIT_MASKWRITE(0xF8000740, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000744, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000748, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF800074C, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000750, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000754, 0x00003FFFU, 0x000007A0U),
	EMIT_MASKWRITE(0xF8000770, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000774, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000778, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF800077C, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000780, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000784, 0x00003FFFU, 0x00000702U),
	EMIT_MASKWRITE(0xF8000788, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF800078C, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000790, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000794, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF8000798, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF800079C, 0x00003FFFU, 0x00000703U),
	EMIT_MASKWRITE(0xF80007A0, 0x00003FFFU, 0x00000720U),
	EMIT_MASKWRITE(0xF80007A4, 0x00003FFFU, 0x00000721U),
	EMIT_MASKWRITE(0xF80007A8, 0x00003FFFU, 0x000007C0U),
	EMIT_MASKWRITE(0xF80007AC, 0x00003FFFU, 0x000007C1U),
	EMIT_MASKWRITE(0xF80007B0, 0x00003FFFU, 0x00000740U),
	EMIT_MASKWRITE(0xF80007B4, 0x00003FFFU, 0x00000740U),
	EMIT_MASKWRITE(0xF80007B8, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007BC, 0x00003FFFU, 0x00000660U),
	EMIT_MASKWRITE(0xF80007C0, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007C4, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007C8, 0x00003FFFU, 0x00000661U),
	EMIT_MASKWRITE(0xF80007CC, 0x00003FFFU, 0x00000660U),
	EMIT_MASKWRITE(0xF80007D0, 0x00003FFFU, 0x000006A0U),
	EMIT_MASKWRITE(0xF80007D4, 0x00003FFFU, 0x000006A0U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long ps7_peripherals_init_data_1_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000B48, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B4C, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B50, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000B54, 0x00000180U, 0x00000180U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_MASKWRITE(0xE0000034, 0x000000FFU, 0x00000006U),
	EMIT_MASKWRITE(0xE0000018, 0x0000FFFFU, 0x0000003EU),
	EMIT_MASKWRITE(0xE0000000, 0x000001FFU, 0x00000017U),
	EMIT_MASKWRITE(0xE0000004, 0x00000FFFU, 0x00000020U),
	EMIT_MASKWRITE(0xE000D000, 0x00080000U, 0x00080000U),
	EMIT_MASKWRITE(0xF8007000, 0x20000000U, 0x00000000U),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_MASKDELAY(0xF8F00200, 1),
	EMIT_EXIT(),
};

static unsigned long ps7_post_config_1_0[] = {
	EMIT_MASKWRITE(0xF8000008, 0x0000FFFFU, 0x0000DF0DU),
	EMIT_MASKWRITE(0xF8000900, 0x0000000FU, 0x0000000FU),
	EMIT_MASKWRITE(0xF8000240, 0xFFFFFFFFU, 0x00000000U),
	EMIT_MASKWRITE(0xF8000004, 0x0000FFFFU, 0x0000767BU),
	EMIT_EXIT(),
};

static unsigned long *ps7_mio_init_data = ps7_mio_init_data_3_0;
static unsigned long *ps7_pll_init_data = ps7_pll_init_data_3_0;
static unsigned long *ps7_clock_init_data = ps7_clock_init_data_3_0;
static unsigned long *ps7_ddr_init_data = ps7_ddr_init_data_3_0;
static unsigned long *ps7_peripherals_init_data = ps7_peripherals_init_data_3_0;

int ps7_post_config(void)
{
	unsigned long si_ver = ps7GetSiliconVersion();
	int ret = -1;

	if (si_ver == PCW_SILICON_VERSION_1) {
		ret = ps7_config(ps7_post_config_1_0);
		if (ret != PS7_INIT_SUCCESS)
			return ret;
	} else if (si_ver == PCW_SILICON_VERSION_2) {
		ret = ps7_config(ps7_post_config_2_0);
		if (ret != PS7_INIT_SUCCESS)
			return ret;
	} else {
		ret = ps7_config(ps7_post_config_3_0);
		if (ret != PS7_INIT_SUCCESS)
			return ret;
	}
	return PS7_INIT_SUCCESS;
}

int ps7_init(void)
{
	unsigned long si_ver = ps7GetSiliconVersion();
	int ret;

	if (si_ver == PCW_SILICON_VERSION_1) {
		ps7_mio_init_data = ps7_mio_init_data_1_0;
		ps7_pll_init_data = ps7_pll_init_data_1_0;
		ps7_clock_init_data = ps7_clock_init_data_1_0;
		ps7_ddr_init_data = ps7_ddr_init_data_1_0;
		ps7_peripherals_init_data = ps7_peripherals_init_data_1_0;

	} else if (si_ver == PCW_SILICON_VERSION_2) {
		ps7_mio_init_data = ps7_mio_init_data_2_0;
		ps7_pll_init_data = ps7_pll_init_data_2_0;
		ps7_clock_init_data = ps7_clock_init_data_2_0;
		ps7_ddr_init_data = ps7_ddr_init_data_2_0;
		ps7_peripherals_init_data = ps7_peripherals_init_data_2_0;

	} else {
		ps7_mio_init_data = ps7_mio_init_data_3_0;
		ps7_pll_init_data = ps7_pll_init_data_3_0;
		ps7_clock_init_data = ps7_clock_init_data_3_0;
		ps7_ddr_init_data = ps7_ddr_init_data_3_0;
		ps7_peripherals_init_data = ps7_peripherals_init_data_3_0;
	}

	ret = ps7_config(ps7_mio_init_data);
	if (ret != PS7_INIT_SUCCESS)
		return ret;

	ret = ps7_config(ps7_pll_init_data);
	if (ret != PS7_INIT_SUCCESS)
		return ret;

	ret = ps7_config(ps7_clock_init_data);
	if (ret != PS7_INIT_SUCCESS)
		return ret;

	ret = ps7_config(ps7_ddr_init_data);
	if (ret != PS7_INIT_SUCCESS)
		return ret;

	ret = ps7_config(ps7_peripherals_init_data);
	if (ret != PS7_INIT_SUCCESS)
		return ret;
	return PS7_INIT_SUCCESS;
}
