/**
 * Waveshare Raspberry Pi Pico 2 W 夹爪控制器
 *
 * 功能: 通过串口接收角度指令 (0-180°)，输出 50Hz PWM (500-2500us)
 * 通讯: 115200 8N1
 *
 * 芯片: RP2350 (Dual-core Cortex-M33)
 * 框架: Arduino-Pico core (earlephilhower/arduino-pico)
 *
 * 引脚: GP2 (Pin 4) 输出 PWM 到舵机
 */

#include <Arduino.h>
#include <hardware/pwm.h>   // Pico SDK PWM
#include <hardware/gpio.h> // Pico SDK GPIO

// ========== 配置 ==========
#ifndef PWM_PIN
#define PWM_PIN 2           // GP2 = Pin 4
#endif

// 舵机脉宽范围 (us)
const int PULSE_MIN_US = 500;   // 0°
const int PULSE_MAX_US = 2500;  // 180°
const float ANGLE_MIN = 0.0f;
const float ANGLE_MAX = 180.0f;

// PWM 参数 (125MHz sys_clk)
const uint PWM_DIVIDER = 38;     // 分频: 125MHz/38 ≈ 3.289MHz
const uint PWM_WRAP = 65535;     // 16-bit 计数器上限

// ========== 全局状态 ==========
float g_current_angle = 90.0f;

// ========== 函数 ==========

/**
 * 角度 -> 脉宽 (us)
 */
int angle_to_pulse_us(float angle) {
    angle = constrain(angle, ANGLE_MIN, ANGLE_MAX);
    float ratio = (angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN);
    return (int)(PULSE_MIN_US + ratio * (PULSE_MAX_US - PULSE_MIN_US));
}

/**
 * 设置夹爪角度
 */
void set_angle(float angle) {
    angle = constrain(angle, ANGLE_MIN, ANGLE_MAX);
    g_current_angle = angle;

    int pulse_us = angle_to_pulse_us(angle);

    // PWM_freq ≈ 50Hz (125MHz / 38 / 65535 ≈ 50.2Hz)
    // level = pulse_us / 20000 * PWM_WRAP  (因为 20000us = 1 PWM period)
    uint32_t level = (uint32_t)((uint64_t)pulse_us * PWM_WRAP / 20000UL);

    uint slice = pwm_gpio_to_slice_num(PWM_PIN);
    pwm_set_gpio_level(PWM_PIN, level);
}

/**
 * 处理串口命令
 * 格式:
 *   G<angle>\n   - 设置角度 (例: G90\n -> 90°)
 *   Q\n          - 查询当前角度
 *   O\n          - 打开夹爪 (180°)
 *   C\n          - 关闭夹爪 (0°)
 */
void handle_command(const String& cmd) {
    String trimmed = cmd;
    trimmed.trim();
    if (trimmed.length() == 0) return;

    char prefix = trimmed.charAt(0);

    switch (prefix) {
        case 'G':
        case 'g': {
            float angle = trimmed.substring(1).toFloat();
            set_angle(angle);
            Serial.printf("OK%.1f\n", g_current_angle);
            break;
        }
        case 'Q':
        case 'q': {
            Serial.printf("A%.1f\n", g_current_angle);
            break;
        }
        case 'O':
        case 'o': {
            set_angle(ANGLE_MAX);
            Serial.printf("OK%.1f\n", g_current_angle);
            break;
        }
        case 'C':
        case 'c': {
            set_angle(ANGLE_MIN);
            Serial.printf("OK%.1f\n", g_current_angle);
            break;
        }
        default:
            Serial.printf("ERR Unknown: %s\n", trimmed.c_str());
            break;
    }
}

/**
 * 初始化 PWM
 */
void init_pwm() {
    // GPIO 复用为 PWM
    gpio_set_function(PWM_PIN, GPIO_FUNC_PWM);

    uint slice = pwm_gpio_to_slice_num(PWM_PIN);

    // 停止 PWM slice
    pwm_set_enabled(slice, false);

    // 设置时钟分频
    pwm_set_clkdiv(slice, (float)PWM_DIVIDER);

    // 设置计数器上限
    pwm_set_wrap(slice, PWM_WRAP);

    // 设置初始 duty
    uint32_t level = (uint32_t)((uint64_t)angle_to_pulse_us(g_current_angle) * PWM_WRAP / 20000UL);
    pwm_set_gpio_level(PWM_PIN, level);

    // 启动 PWM
    pwm_set_enabled(slice, true);
}

// ========== Arduino 生命周期 ==========

void setup() {
    Serial.begin(115200);
    // 等待串口连接
    unsigned long start = millis();
    while (!Serial && millis() - start < 3000) {
        delay(10);
    }

    init_pwm();

    Serial.println("Pico 2 W Gripper Ready");
    Serial.println("Cmds: G<0-180>, Q=query, O=open, C=close");
}

void loop() {
    while (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        handle_command(line);
    }
    delay(1);
}
