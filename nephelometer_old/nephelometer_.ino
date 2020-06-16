/* 
    Nephelometer Steuerung
    Bachelorprojekt Brinkmann, Eberhard, Stock 2020
*/

// Pins
#define RESET 7
#define PULSE 8
#define SENSOR A1

// Integrationszeit
#define T 100

void setup() {
    pinMode(RESET, OUTPUT);    // Pin for discharging the capacitor
    pinMode(PULSE, OUTPUT);    // Pin for turning the LED on/off
    Serial.begin(9600);
    Serial.println("  Timestamp | Background value | Measured value | Final value ");
    Serial.println("------------+------------------+----------------+-------------");
}

int record(bool led_on) {
    digitalWrite(RESET, HIGH);        // Reset capacitor (close bypass connection)
    delayMicroseconds(100);           // Let capacitor decharge (R = 100 Ohm, C = 100 nF => tau = 10 us)
    if (led_on)
        digitalWrite(PULSE, HIGH);    // Turn LED on!
    digitalWrite(RESET, LOW);         // Start integration (open  bypass connection)
    delay(T);                         // Integrate for T milliseconds
    if (led_on)
      digitalWrite(PULSE, LOW);       // Turn LED off!
    return analogRead(SENSOR);        // Record sensor measurement
}

void loop() {
    // Record time to adjust delay for next measurement
    unsigned long time_start = millis();

    // Make the measurements
    int led_off = record(false);    // Background measurement
    int led_on  = record(true);     // Measurement with LED on

    // Final measured value (normally between 0 and 1, if negative then background (LED off) > measurement (LED on))
    float value = (val - background) / 1024.0;

    // Print result to Serial
    char str[64];
    char value_str[10];
    dtostrf(value, 7, 4, value_str);
    sprintf(str, " %010lu | %16d | %14d | %11s", time_start, led_off, led_on, value_str);
    Serial.println(str);
    
    // Repeat exactly every 500ms
    unsigned long time_diff = (millis() - time_start);
    delay(500 - time_diff);
}
