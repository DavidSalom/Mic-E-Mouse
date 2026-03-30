#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <condition_variable>
#include <algorithm>
#include <atomic>
#include <signal.h>
#include <cstring>
#include <cstdio>
#include "common.h"

enum class MOUSE_TYPE { MOUSE_8K, MOUSE_4K, NONE };

// Configuración de tasas de muestreo
static std::map<MOUSE_TYPE, int> poll_map = { 
    {MOUSE_TYPE::MOUSE_8K, 8000}, 
    {MOUSE_TYPE::MOUSE_4K, 4000} 
};

// Variables globales corregidas
static int SAMPLE_RATE = 8000;
static MOUSE_TYPE m_type = MOUSE_TYPE::NONE;
static std::string mouse_cmd = "./elegir_mouse.sh";
static std::string cli_filename = "";
static std::atomic<bool> running(true); 
static mouse_t* global_mouse = nullptr;

#define COLLECTION_TIME 30

// Estructura de Buffer mejorada
struct buffer_t {
    entry_t* buffer;
    int head = 0;
    int max_size;
    std::mutex mtx;
    std::condition_variable cv;
    bool sanity = false;

    buffer_t(int size) : max_size(size) {
        buffer = new entry_t[size];
    }
    ~buffer_t() { delete[] buffer; }

    bool full() { return head >= max_size; }

    bool push(entry_t entry) {
        std::unique_lock<std::mutex> lock(mtx);
        if (head < max_size) {
            buffer[head++] = entry;
            if (full()) {
                cv.notify_one();
                return true;
            }
        }
        return false;
    }

    void stringify(char* str) {
        std::unique_lock<std::mutex> lock(mtx);
        str[0] = '\0';
        int offset = 0;
        for (int i = 0; i < head; i++) {
            offset += sprintf(str + offset, "%d, %d, %d\n", buffer[i].time, buffer[i].x, buffer[i].y);
        }
        sanity = true;
    }

    void reset() {
        std::unique_lock<std::mutex> lock(mtx);
        head = 0;
        sanity = false;
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return full() || !running; });
    }
};

buffer_t* A = nullptr;
buffer_t* B = nullptr;

// Captura la salida del script (stdout) mientras el menú sale por stderr
std::string get_command_output(std::string cmd) {
    std::string result = "";
    char buffer[128];
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    pclose(pipe);
    result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
    result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    return result;
}

void interrupt_handler(int signum) {
    running = false;
    printf("\n[SIGNAL] Interrupt (%d) recibida. Finalizando...\n", signum);
    if (A) A->cv.notify_all();
    if (B) B->cv.notify_all();
}

void collector_thread() {
    std::string mouse_node = get_command_output(mouse_cmd);
    if (mouse_node.empty()) {
        printf("Error: No se recibió un nodo válido del script.\n");
        running = false;
        return;
    }

    std::string mouse_device = "/dev/input/" + mouse_node;
    printf("Usando dispositivo: %s\n", mouse_device.c_str());

    mouse_t mouse(const_cast<char*>(mouse_device.c_str()));
    global_mouse = &mouse;
    stopwatch_t stopwatch;
    
    buffer_t* current_buffer = A;
    buffer_t* other_buffer = B;

    mouse.getData();
    stopwatch.click();

    while (running) {
        mouse.getData();
        stopwatch.click();
        entry_t entry = {mouse.x, mouse.y, (int)stopwatch.refreshTime()};
        
        if (current_buffer->push(entry)) {
            printf("[COLLECTOR]: Buffer lleno, cambiando...\n");
            std::swap(current_buffer, other_buffer);
        }
    }
    printf("[COLLECTOR]: Salida.\n");
}

void archiver_thread() {
    time_t now = time(0);
    tm* ltm = localtime(&now);
    char date[180];
    strftime(date, 180, "%Y-%m-%d_%H-%M-%S", ltm);
    char filename[256];

    if (cli_filename.empty() || cli_filename == "NULL") {
        sprintf(filename, "./data/data_%s.csv", date);
    } else {
        strncpy(filename, cli_filename.c_str(), 255);
    }

    file_handler file(filename);
    buffer_t* current_buffer = A;
    buffer_t* other_buffer = B;
    char* str = new char[(SAMPLE_RATE * COLLECTION_TIME) * 64];

    file.write((char*)"offset, x, y\n");

    while (running) {
        current_buffer->wait();
        if (!running && !current_buffer->full()) break;

        printf("[ARCHIVER]: Escribiendo a disco...\n");
        current_buffer->stringify(str);
        file.write(str);
        current_buffer->reset();
        std::swap(current_buffer, other_buffer);
    }

    printf("[ARCHIVER]: Volcado final...\n");
    current_buffer->stringify(str);
    file.write(str);
    delete[] str;
    printf("[ARCHIVER]: Salida.\n");
}

int main(int argc, char** argv) {
    if (argc >= 2) cli_filename = argv[1];

    // Selección de hardware
    if (argc == 3) {
        m_type = static_cast<MOUSE_TYPE>(std::stoi(argv[2]));
    } else {
        std::cout << "[0]: Mouse (8KHz)\n[1]: Mouse (4KHz)\nElija ratón: ";
        int sel; std::cin >> sel;
        if (sel < 0 || sel >= (int)MOUSE_TYPE::NONE) return 1;
        m_type = static_cast<MOUSE_TYPE>(sel);
    }

    // Inicialización dinámica según selección
    SAMPLE_RATE = poll_map[m_type];
    int buf_size = SAMPLE_RATE * COLLECTION_TIME;
    A = new buffer_t(buf_size);
    B = new buffer_t(buf_size);

    signal(SIGINT, interrupt_handler);

    std::thread collector(collector_thread);
    std::thread archiver(archiver_thread);

    printf("Ejecutando captura...\n");
    archiver.join();

    if (global_mouse) global_mouse->endData();
    collector.join();

    delete A; delete B;
    printf("Finalizado correctamente.\n");
    return 0;
}
