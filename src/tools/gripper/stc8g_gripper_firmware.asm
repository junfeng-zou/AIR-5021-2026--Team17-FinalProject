;--------------------------------------------------------
; File Created by SDCC : free open source ANSI-C Compiler
; Version 4.0.0 #11528 (Linux)
;--------------------------------------------------------
	.module stc8g_gripper_firmware
	.optsdcc -mmcs51 --model-small
	
;--------------------------------------------------------
; Public variables in this module
;--------------------------------------------------------
	.globl _main
	.globl _set_angle
	.globl _sendstr
	.globl _send
	.globl _uart_init
	.globl _timer0_init
	.globl _delay
	.globl _CY
	.globl _AC
	.globl _F0
	.globl _RS1
	.globl _RS0
	.globl _OV
	.globl _F1
	.globl _P
	.globl _PS
	.globl _PT1
	.globl _PX1
	.globl _PT0
	.globl _PX0
	.globl _RD
	.globl _WR
	.globl _T1
	.globl _T0
	.globl _INT1
	.globl _INT0
	.globl _TXD
	.globl _RXD
	.globl _P3_7
	.globl _P3_6
	.globl _P3_5
	.globl _P3_4
	.globl _P3_3
	.globl _P3_2
	.globl _P3_1
	.globl _P3_0
	.globl _EA
	.globl _ES
	.globl _ET1
	.globl _EX1
	.globl _ET0
	.globl _EX0
	.globl _P2_7
	.globl _P2_6
	.globl _P2_5
	.globl _P2_4
	.globl _P2_3
	.globl _P2_2
	.globl _P2_1
	.globl _P2_0
	.globl _SM0
	.globl _SM1
	.globl _SM2
	.globl _REN
	.globl _TB8
	.globl _RB8
	.globl _TI
	.globl _RI
	.globl _P1_7
	.globl _P1_6
	.globl _P1_5
	.globl _P1_4
	.globl _P1_3
	.globl _P1_2
	.globl _P1_1
	.globl _P1_0
	.globl _TF1
	.globl _TR1
	.globl _TF0
	.globl _TR0
	.globl _IE1
	.globl _IT1
	.globl _IE0
	.globl _IT0
	.globl _P0_7
	.globl _P0_6
	.globl _P0_5
	.globl _P0_4
	.globl _P0_3
	.globl _P0_2
	.globl _P0_1
	.globl _P0_0
	.globl _P3M1
	.globl _P3M0
	.globl _B
	.globl _ACC
	.globl _PSW
	.globl _IP
	.globl _P3
	.globl _IE
	.globl _P2
	.globl _SBUF
	.globl _SCON
	.globl _P1
	.globl _TH1
	.globl _TH0
	.globl _TL1
	.globl _TL0
	.globl _TMOD
	.globl _TCON
	.globl _PCON
	.globl _DPH
	.globl _DPL
	.globl _SP
	.globl _P0
	.globl _g_pwm_duty
	.globl _g_uart_ready
	.globl _g_uart_idx
	.globl _g_uart_buf
	.globl _g_angle
	.globl _Timer0_ISR
	.globl _UART_ISR
;--------------------------------------------------------
; special function registers
;--------------------------------------------------------
	.area RSEG    (ABS,DATA)
	.org 0x0000
_P0	=	0x0080
_SP	=	0x0081
_DPL	=	0x0082
_DPH	=	0x0083
_PCON	=	0x0087
_TCON	=	0x0088
_TMOD	=	0x0089
_TL0	=	0x008a
_TL1	=	0x008b
_TH0	=	0x008c
_TH1	=	0x008d
_P1	=	0x0090
_SCON	=	0x0098
_SBUF	=	0x0099
_P2	=	0x00a0
_IE	=	0x00a8
_P3	=	0x00b0
_IP	=	0x00b8
_PSW	=	0x00d0
_ACC	=	0x00e0
_B	=	0x00f0
_P3M0	=	0x00c4
_P3M1	=	0x00f3
;--------------------------------------------------------
; special function bits
;--------------------------------------------------------
	.area RSEG    (ABS,DATA)
	.org 0x0000
_P0_0	=	0x0080
_P0_1	=	0x0081
_P0_2	=	0x0082
_P0_3	=	0x0083
_P0_4	=	0x0084
_P0_5	=	0x0085
_P0_6	=	0x0086
_P0_7	=	0x0087
_IT0	=	0x0088
_IE0	=	0x0089
_IT1	=	0x008a
_IE1	=	0x008b
_TR0	=	0x008c
_TF0	=	0x008d
_TR1	=	0x008e
_TF1	=	0x008f
_P1_0	=	0x0090
_P1_1	=	0x0091
_P1_2	=	0x0092
_P1_3	=	0x0093
_P1_4	=	0x0094
_P1_5	=	0x0095
_P1_6	=	0x0096
_P1_7	=	0x0097
_RI	=	0x0098
_TI	=	0x0099
_RB8	=	0x009a
_TB8	=	0x009b
_REN	=	0x009c
_SM2	=	0x009d
_SM1	=	0x009e
_SM0	=	0x009f
_P2_0	=	0x00a0
_P2_1	=	0x00a1
_P2_2	=	0x00a2
_P2_3	=	0x00a3
_P2_4	=	0x00a4
_P2_5	=	0x00a5
_P2_6	=	0x00a6
_P2_7	=	0x00a7
_EX0	=	0x00a8
_ET0	=	0x00a9
_EX1	=	0x00aa
_ET1	=	0x00ab
_ES	=	0x00ac
_EA	=	0x00af
_P3_0	=	0x00b0
_P3_1	=	0x00b1
_P3_2	=	0x00b2
_P3_3	=	0x00b3
_P3_4	=	0x00b4
_P3_5	=	0x00b5
_P3_6	=	0x00b6
_P3_7	=	0x00b7
_RXD	=	0x00b0
_TXD	=	0x00b1
_INT0	=	0x00b2
_INT1	=	0x00b3
_T0	=	0x00b4
_T1	=	0x00b5
_WR	=	0x00b6
_RD	=	0x00b7
_PX0	=	0x00b8
_PT0	=	0x00b9
_PX1	=	0x00ba
_PT1	=	0x00bb
_PS	=	0x00bc
_P	=	0x00d0
_F1	=	0x00d1
_OV	=	0x00d2
_RS0	=	0x00d3
_RS1	=	0x00d4
_F0	=	0x00d5
_AC	=	0x00d6
_CY	=	0x00d7
;--------------------------------------------------------
; overlayable register banks
;--------------------------------------------------------
	.area REG_BANK_0	(REL,OVR,DATA)
	.ds 8
;--------------------------------------------------------
; internal ram data
;--------------------------------------------------------
	.area DSEG    (DATA)
_g_angle::
	.ds 1
_g_uart_buf::
	.ds 16
_g_uart_idx::
	.ds 1
_g_uart_ready::
	.ds 1
_g_pwm_duty::
	.ds 2
_Timer0_ISR_tick_65536_21:
	.ds 2
;--------------------------------------------------------
; overlayable items in internal ram 
;--------------------------------------------------------
;--------------------------------------------------------
; Stack segment in internal ram 
;--------------------------------------------------------
	.area	SSEG
__start__stack:
	.ds	1

;--------------------------------------------------------
; indirectly addressable internal ram data
;--------------------------------------------------------
	.area ISEG    (DATA)
;--------------------------------------------------------
; absolute internal ram data
;--------------------------------------------------------
	.area IABS    (ABS,DATA)
	.area IABS    (ABS,DATA)
;--------------------------------------------------------
; bit data
;--------------------------------------------------------
	.area BSEG    (BIT)
;--------------------------------------------------------
; paged external ram data
;--------------------------------------------------------
	.area PSEG    (PAG,XDATA)
;--------------------------------------------------------
; external ram data
;--------------------------------------------------------
	.area XSEG    (XDATA)
;--------------------------------------------------------
; absolute external ram data
;--------------------------------------------------------
	.area XABS    (ABS,XDATA)
;--------------------------------------------------------
; external initialized ram data
;--------------------------------------------------------
	.area XISEG   (XDATA)
	.area HOME    (CODE)
	.area GSINIT0 (CODE)
	.area GSINIT1 (CODE)
	.area GSINIT2 (CODE)
	.area GSINIT3 (CODE)
	.area GSINIT4 (CODE)
	.area GSINIT5 (CODE)
	.area GSINIT  (CODE)
	.area GSFINAL (CODE)
	.area CSEG    (CODE)
;--------------------------------------------------------
; interrupt vector 
;--------------------------------------------------------
	.area HOME    (CODE)
__interrupt_vect:
	ljmp	__sdcc_gsinit_startup
	reti
	.ds	7
	ljmp	_Timer0_ISR
	.ds	5
	reti
	.ds	7
	reti
	.ds	7
	ljmp	_UART_ISR
;--------------------------------------------------------
; global & static initialisations
;--------------------------------------------------------
	.area HOME    (CODE)
	.area GSINIT  (CODE)
	.area GSFINAL (CODE)
	.area GSINIT  (CODE)
	.globl __sdcc_gsinit_startup
	.globl __sdcc_program_startup
	.globl __start__stack
	.globl __mcs51_genXINIT
	.globl __mcs51_genXRAMCLEAR
	.globl __mcs51_genRAMCLEAR
;------------------------------------------------------------
;Allocation info for local variables in function 'Timer0_ISR'
;------------------------------------------------------------
;tick                      Allocated with name '_Timer0_ISR_tick_65536_21'
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:110: static unsigned int tick = 0;
	mov	a,#0x00
	mov	_Timer0_ISR_tick_65536_21,a
	mov	(_Timer0_ISR_tick_65536_21 + 1),a
;	stc8g_gripper_firmware.c:16: volatile unsigned char g_angle = 90;
	mov	_g_angle,#0x5a
;	stc8g_gripper_firmware.c:18: volatile unsigned char g_uart_idx = 0;
	mov	_g_uart_idx,#0x00
;	stc8g_gripper_firmware.c:19: volatile unsigned char g_uart_ready = 0;
	mov	_g_uart_ready,#0x00
;	stc8g_gripper_firmware.c:20: volatile unsigned int g_pwm_duty = 1500;
	mov	_g_pwm_duty,#0xdc
	mov	(_g_pwm_duty + 1),#0x05
	.area GSFINAL (CODE)
	ljmp	__sdcc_program_startup
;--------------------------------------------------------
; Home
;--------------------------------------------------------
	.area HOME    (CODE)
	.area HOME    (CODE)
__sdcc_program_startup:
	ljmp	_main
;	return from main will return to caller
;--------------------------------------------------------
; code
;--------------------------------------------------------
	.area CSEG    (CODE)
;------------------------------------------------------------
;Allocation info for local variables in function 'delay'
;------------------------------------------------------------
;x                         Allocated to stack - _bp +1
;i                         Allocated to registers r4 r5 
;j                         Allocated to registers r2 r3 
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:23: void delay(unsigned int x)
;	-----------------------------------------
;	 function delay
;	-----------------------------------------
_delay:
	ar7 = 0x07
	ar6 = 0x06
	ar5 = 0x05
	ar4 = 0x04
	ar3 = 0x03
	ar2 = 0x02
	ar1 = 0x01
	ar0 = 0x00
	push	_bp
	mov	_bp,sp
	push	dpl
	push	dph
;	stc8g_gripper_firmware.c:26: for (i = 0; i < x; i++)
	mov	r4,#0x00
	mov	r5,#0x00
00107$:
	mov	r0,_bp
	inc	r0
	clr	c
	mov	a,r4
	subb	a,@r0
	mov	a,r5
	inc	r0
	subb	a,@r0
	jc	00128$
	ljmp	00109$
00128$:
;	stc8g_gripper_firmware.c:27: for (j = 0; j < 600; j++);
	mov	r2,#0x58
	mov	r3,#0x02
00105$:
	mov	a,r2
	add	a,#0xff
	mov	r6,a
	mov	a,r3
	addc	a,#0xff
	mov	r7,a
	mov	ar2,r6
	mov	ar3,r7
	mov	a,r6
	orl	a,r7
	jz	00129$
	ljmp	00105$
00129$:
;	stc8g_gripper_firmware.c:26: for (i = 0; i < x; i++)
	inc	r4
	cjne	r4,#0x00,00130$
	inc	r5
00130$:
	ljmp	00107$
00109$:
;	stc8g_gripper_firmware.c:28: }
	mov	sp,_bp
	pop	_bp
	ret
;------------------------------------------------------------
;Allocation info for local variables in function 'timer0_init'
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:31: void timer0_init(void)
;	-----------------------------------------
;	 function timer0_init
;	-----------------------------------------
_timer0_init:
;	stc8g_gripper_firmware.c:33: TMOD = 0x01;  // 16位模式
	mov	_TMOD,#0x01
;	stc8g_gripper_firmware.c:34: TH0 = 0xFC;   // 1ms 溢出
	mov	_TH0,#0xfc
;	stc8g_gripper_firmware.c:35: TL0 = 0x66;
	mov	_TL0,#0x66
;	stc8g_gripper_firmware.c:36: ET0 = 1;       // 开中断
;	assignBit
	setb	_ET0
;	stc8g_gripper_firmware.c:37: TR0 = 1;       // 启动
;	assignBit
	setb	_TR0
00101$:
;	stc8g_gripper_firmware.c:38: }
	ret
;------------------------------------------------------------
;Allocation info for local variables in function 'uart_init'
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:41: void uart_init(void)
;	-----------------------------------------
;	 function uart_init
;	-----------------------------------------
_uart_init:
;	stc8g_gripper_firmware.c:43: SCON = 0x50;
	mov	_SCON,#0x50
;	stc8g_gripper_firmware.c:44: TMOD |= 0x20;
	orl	_TMOD,#0x20
;	stc8g_gripper_firmware.c:45: TH1 = 0xFA;
	mov	_TH1,#0xfa
;	stc8g_gripper_firmware.c:46: TL1 = 0xFA;
	mov	_TL1,#0xfa
;	stc8g_gripper_firmware.c:47: PCON |= 0x80;
	orl	_PCON,#0x80
;	stc8g_gripper_firmware.c:48: TR1 = 1;
;	assignBit
	setb	_TR1
;	stc8g_gripper_firmware.c:49: ES = 1;
;	assignBit
	setb	_ES
00101$:
;	stc8g_gripper_firmware.c:50: }
	ret
;------------------------------------------------------------
;Allocation info for local variables in function 'send'
;------------------------------------------------------------
;c                         Allocated to registers 
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:53: void send(char c)
;	-----------------------------------------
;	 function send
;	-----------------------------------------
_send:
	mov	_SBUF,dpl
;	stc8g_gripper_firmware.c:56: while (!TI);
00101$:
	jb	_TI,00114$
	ljmp	00101$
00114$:
;	stc8g_gripper_firmware.c:57: TI = 0;
;	assignBit
	clr	_TI
00104$:
;	stc8g_gripper_firmware.c:58: }
	ret
;------------------------------------------------------------
;Allocation info for local variables in function 'sendstr'
;------------------------------------------------------------
;s                         Allocated to registers 
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:61: void sendstr(const char *s)
;	-----------------------------------------
;	 function sendstr
;	-----------------------------------------
_sendstr:
	mov	r5,dpl
	mov	r6,dph
	mov	r7,b
;	stc8g_gripper_firmware.c:63: while (*s) send(*s++);
00101$:
	mov	dpl,r5
	mov	dph,r6
	mov	b,r7
	lcall	__gptrget
	mov	r4,a
	mov	a,r4
	jnz	00115$
	ljmp	00104$
00115$:
	inc	r5
	cjne	r5,#0x00,00116$
	inc	r6
00116$:
	mov	dpl,r4
	push	ar7
	push	ar6
	push	ar5
	lcall	_send
	pop	ar5
	pop	ar6
	pop	ar7
	ljmp	00101$
00104$:
;	stc8g_gripper_firmware.c:64: }
	ret
;------------------------------------------------------------
;Allocation info for local variables in function 'set_angle'
;------------------------------------------------------------
;a                         Allocated to registers r7 
;duty                      Allocated to registers 
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:67: void set_angle(unsigned char a)
;	-----------------------------------------
;	 function set_angle
;	-----------------------------------------
_set_angle:
	mov	r7,dpl
;	stc8g_gripper_firmware.c:70: if (a > 180) a = 180;
	clr	c
	mov	a,#0xb4
	subb	a,r7
	jc	00109$
	ljmp	00102$
00109$:
	mov	r7,#0xb4
00102$:
;	stc8g_gripper_firmware.c:71: g_angle = a;
	mov	_g_angle,r7
;	stc8g_gripper_firmware.c:72: duty = 500 + ((unsigned int)a * 2000) / 180;
	mov	r6,#0x00
	push	ar7
	push	ar6
	mov	dpl,#0xd0
	mov	dph,#0x07
	lcall	__mulint
	dec	sp
	dec	sp
	mov	a,#0xb4
	push	acc
	mov	a,#0x00
	push	acc
	lcall	__divuint
	mov	r6,dpl
	mov	r7,dph
	dec	sp
	dec	sp
	mov	a,#0xf4
	add	a,r6
	mov	_g_pwm_duty,a
	mov	a,#0x01
	addc	a,r7
	mov	(_g_pwm_duty + 1),a
;	stc8g_gripper_firmware.c:73: g_pwm_duty = duty;
00103$:
;	stc8g_gripper_firmware.c:74: }
	ret
;------------------------------------------------------------
;Allocation info for local variables in function 'main'
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:77: void main(void)
;	-----------------------------------------
;	 function main
;	-----------------------------------------
_main:
;	stc8g_gripper_firmware.c:80: P3M0 |= 0x08;
	orl	_P3M0,#0x08
;	stc8g_gripper_firmware.c:81: P3M1 &= ~0x08;
	anl	_P3M1,#0xf7
;	stc8g_gripper_firmware.c:84: uart_init();
	lcall	_uart_init
;	stc8g_gripper_firmware.c:85: timer0_init();
	lcall	_timer0_init
;	stc8g_gripper_firmware.c:86: EA = 1;  // 开总中断
;	assignBit
	setb	_EA
;	stc8g_gripper_firmware.c:89: P3_3 = 0;
;	assignBit
	clr	_P3_3
;	stc8g_gripper_firmware.c:90: delay(1000);
	mov	dpl,#0xe8
	mov	dph,#0x03
	lcall	_delay
;	stc8g_gripper_firmware.c:91: P3_3 = 1;
;	assignBit
	setb	_P3_3
;	stc8g_gripper_firmware.c:92: delay(1000);
	mov	dpl,#0xe8
	mov	dph,#0x03
	lcall	_delay
;	stc8g_gripper_firmware.c:93: P3_3 = 0;
;	assignBit
	clr	_P3_3
;	stc8g_gripper_firmware.c:94: delay(1000);
	mov	dpl,#0xe8
	mov	dph,#0x03
	lcall	_delay
;	stc8g_gripper_firmware.c:95: P3_3 = 1;
;	assignBit
	setb	_P3_3
;	stc8g_gripper_firmware.c:98: sendstr("START\n");
	mov	dpl,#___str_0
	mov	dph,#(___str_0 >> 8)
	mov	b,#0x80
	lcall	_sendstr
;	stc8g_gripper_firmware.c:101: while (1) {
00102$:
;	stc8g_gripper_firmware.c:102: P3_3 = !P3_3;
	cpl	_P3_3
;	stc8g_gripper_firmware.c:103: delay(500);
	mov	dpl,#0xf4
	mov	dph,#0x01
	lcall	_delay
	ljmp	00102$
00104$:
;	stc8g_gripper_firmware.c:105: }
	ret
;------------------------------------------------------------
;Allocation info for local variables in function 'Timer0_ISR'
;------------------------------------------------------------
;tick                      Allocated with name '_Timer0_ISR_tick_65536_21'
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:108: void Timer0_ISR(void) __interrupt(1)
;	-----------------------------------------
;	 function Timer0_ISR
;	-----------------------------------------
_Timer0_ISR:
	push	acc
	push	b
	push	dpl
	push	dph
	push	psw
	mov	psw,#0x00
;	stc8g_gripper_firmware.c:113: TH0 = 0xFC;
	mov	_TH0,#0xfc
;	stc8g_gripper_firmware.c:114: TL0 = 0x66;
	mov	_TL0,#0x66
;	stc8g_gripper_firmware.c:116: tick++;
	inc	_Timer0_ISR_tick_65536_21
	clr	a
	cjne	a,_Timer0_ISR_tick_65536_21,00115$
	inc	(_Timer0_ISR_tick_65536_21 + 1)
00115$:
;	stc8g_gripper_firmware.c:117: if (tick >= 20000) {  // 20ms
	clr	c
	mov	a,_Timer0_ISR_tick_65536_21
	subb	a,#0x20
	mov	a,(_Timer0_ISR_tick_65536_21 + 1)
	subb	a,#0x4e
	jnc	00116$
	ljmp	00102$
00116$:
;	stc8g_gripper_firmware.c:118: tick = 0;
	mov	a,#0x00
	mov	_Timer0_ISR_tick_65536_21,a
	mov	(_Timer0_ISR_tick_65536_21 + 1),a
;	stc8g_gripper_firmware.c:119: P3_3 = 1;
;	assignBit
	setb	_P3_3
00102$:
;	stc8g_gripper_firmware.c:121: if (tick >= g_pwm_duty) {
	clr	c
	mov	a,_Timer0_ISR_tick_65536_21
	subb	a,_g_pwm_duty
	mov	a,(_Timer0_ISR_tick_65536_21 + 1)
	subb	a,(_g_pwm_duty + 1)
	jnc	00117$
	ljmp	00105$
00117$:
;	stc8g_gripper_firmware.c:122: P3_3 = 0;
;	assignBit
	clr	_P3_3
00105$:
;	stc8g_gripper_firmware.c:124: }
	pop	psw
	pop	dph
	pop	dpl
	pop	b
	pop	acc
	reti
;------------------------------------------------------------
;Allocation info for local variables in function 'UART_ISR'
;------------------------------------------------------------
;c                         Allocated to registers r7 
;------------------------------------------------------------
;	stc8g_gripper_firmware.c:127: void UART_ISR(void) __interrupt(4)
;	-----------------------------------------
;	 function UART_ISR
;	-----------------------------------------
_UART_ISR:
	push	acc
	push	b
	push	dpl
	push	dph
	push	ar7
	push	ar6
	push	ar1
	push	ar0
	push	psw
	mov	psw,#0x00
;	stc8g_gripper_firmware.c:131: if (RI) {
	jb	_RI,00129$
	ljmp	00111$
00129$:
;	stc8g_gripper_firmware.c:132: RI = 0;
;	assignBit
	clr	_RI
;	stc8g_gripper_firmware.c:133: c = SBUF;
	mov	r7,_SBUF
;	stc8g_gripper_firmware.c:135: if (c == '\n' || c == '\r') {
	cjne	r7,#0x0a,00130$
	ljmp	00105$
00130$:
	cjne	r7,#0x0d,00131$
	sjmp	00132$
00131$:
	ljmp	00106$
00132$:
00105$:
;	stc8g_gripper_firmware.c:136: if (g_uart_idx > 0) {
	mov	a,_g_uart_idx
	jnz	00133$
	ljmp	00102$
00133$:
;	stc8g_gripper_firmware.c:137: g_uart_buf[g_uart_idx] = '\0';
	mov	a,_g_uart_idx
	add	a,#_g_uart_buf
	mov	r0,acc
	mov	@r0,#0x00
;	stc8g_gripper_firmware.c:138: g_uart_ready = 1;
	mov	_g_uart_ready,#0x01
00102$:
;	stc8g_gripper_firmware.c:140: g_uart_idx = 0;
	mov	_g_uart_idx,#0x00
	ljmp	00111$
00106$:
;	stc8g_gripper_firmware.c:141: } else if (g_uart_idx < 15) {
	clr	c
	mov	a,_g_uart_idx
	subb	a,#0x0f
	jc	00134$
	ljmp	00111$
00134$:
;	stc8g_gripper_firmware.c:142: g_uart_buf[g_uart_idx++] = c;
	mov	r6,_g_uart_idx
	mov	a,r6
	inc	a
	mov	_g_uart_idx,a
	mov	a,r6
	add	a,#_g_uart_buf
	mov	r0,acc
	mov	@r0,ar7
00111$:
;	stc8g_gripper_firmware.c:145: }
	pop	psw
	pop	ar0
	pop	ar1
	pop	ar6
	pop	ar7
	pop	dph
	pop	dpl
	pop	b
	pop	acc
	reti
	.area CSEG    (CODE)
	.area CONST   (CODE)
	.area CONST   (CODE)
___str_0:
	.ascii "START"
	.db 0x0a
	.db 0x00
	.area CSEG    (CODE)
	.area XINIT   (CODE)
	.area CABS    (ABS,CODE)
