/**
 * STC8G1K08A - 使用标准库启动代码
 */

#include <8051.h>

// STC8G 端口模式寄存器
__sfr __at(0xC4) P3M0;
__sfr __at(0xF3) P3M1;

// 声明中断函数
void UART_ISR(void) __interrupt(4);
void Timer0_ISR(void) __interrupt(1);

// 全局变量
volatile unsigned char g_angle = 90;
volatile unsigned char g_uart_buf[16];
volatile unsigned char g_uart_idx = 0;
volatile unsigned char g_uart_ready = 0;
volatile unsigned int g_pwm_duty = 1500;

// 延时
void delay(unsigned int x)
{
    unsigned int i, j;
    for (i = 0; i < x; i++)
        for (j = 0; j < 600; j++);
}

// 初始化定时器0 - 16位模式
void timer0_init(void)
{
    TMOD = 0x01;  // 16位模式
    TH0 = 0xFC;   // 1ms 溢出
    TL0 = 0x66;
    ET0 = 1;       // 开中断
    TR0 = 1;       // 启动
}

// 初始化串口 - 9600
void uart_init(void)
{
    SCON = 0x50;
    TMOD |= 0x20;
    TH1 = 0xFA;
    TL1 = 0xFA;
    PCON |= 0x80;
    TR1 = 1;
    ES = 1;
}

// 发送字符
void send(char c)
{
    SBUF = c;
    while (!TI);
    TI = 0;
}

// 发送字符串
void sendstr(const char *s)
{
    while (*s) send(*s++);
}

// 设置角度
void set_angle(unsigned char a)
{
    unsigned int duty;
    if (a > 180) a = 180;
    g_angle = a;
    duty = 500 + ((unsigned int)a * 2000) / 180;
    g_pwm_duty = duty;
}

// 主函数
void main(void)
{
    // P3.3 推挽输出
    P3M0 |= 0x08;
    P3M1 &= ~0x08;

    // 初始化
    uart_init();
    timer0_init();
    EA = 1;  // 开总中断

    // 测试：P3.3 闪烁
    P3_3 = 0;
    delay(1000);
    P3_3 = 1;
    delay(1000);
    P3_3 = 0;
    delay(1000);
    P3_3 = 1;

    // 发送启动信息
    sendstr("START\n");

    // 循环
    while (1) {
        P3_3 = !P3_3;
        delay(500);
    }
}

// 定时器0中断 - PWM
void Timer0_ISR(void) __interrupt(1)
{
    static unsigned int tick = 0;

    // 重装定时值
    TH0 = 0xFC;
    TL0 = 0x66;

    tick++;
    if (tick >= 20000) {  // 20ms
        tick = 0;
        P3_3 = 1;
    }
    if (tick >= g_pwm_duty) {
        P3_3 = 0;
    }
}

// 串口中断
void UART_ISR(void) __interrupt(4)
{
    unsigned char c;

    if (RI) {
        RI = 0;
        c = SBUF;

        if (c == '\n' || c == '\r') {
            if (g_uart_idx > 0) {
                g_uart_buf[g_uart_idx] = '\0';
                g_uart_ready = 1;
            }
            g_uart_idx = 0;
        } else if (g_uart_idx < 15) {
            g_uart_buf[g_uart_idx++] = c;
        }
    }
}
