// #define RESET_FLAG 1

#include <ostream>
#include <random>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <thread>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <map>
#include <condition_variable>
#include "common.h"
#include <signal.h>

enum class MOUSE_TYPE {
  RAZER8KHZ,
  G502,
  NONE
};

static std::map<MOUSE_TYPE, int> poll_map = { 
                                              {MOUSE_TYPE::RAZER8KHZ, 8000},
                                              {MOUSE_TYPE::G502     , 1000},    
                                            };

static int SAMPLE_RATE = 8000;
static MOUSE_TYPE m_type = MOUSE_TYPE::RAZER8KHZ;
static std::string mouse_cmd;

#define COLLECTION_TIME 5
#define BUFFER_SIZE (SAMPLE_RATE * COLLECTION_TIME)


static std::string cli_filename = "";
static mouse_t* global_mouse = nullptr;

struct buffer_t{
    entry_t* buffer = new entry_t[BUFFER_SIZE];
    int head = 0;
    bool full(){
        return head == BUFFER_SIZE;
    }
    std::mutex* mutex = new std::mutex();
    std::condition_variable* cv = new std::condition_variable();

    bool push(entry_t entry){
        std::unique_lock<std::mutex> lock(*mutex);
        buffer[head] = entry;
        head++;
        if(full()){
            cv->notify_one();
            return true;
        }
        return false;
    }

    void stringify(char* str){
        std::unique_lock<std::mutex> lock(*mutex);
        for (int i = 0; i < head; i++){
            sprintf(str + strlen(str), "%d, %d, %d\n", buffer[i].time, buffer[i].x, buffer[i].y);
        }
    }

    void reset(){
        std::unique_lock<std::mutex> lock(*mutex);
        head = 0;
    }

    void wait(){
        std::unique_lock<std::mutex> lock(*mutex);
        cv->wait(lock);
    }
};

buffer_t* A = new buffer_t();
buffer_t* B = new buffer_t();

bool* running = new bool(true);

void collector_thread(){

    char buffer_cmd[128];
    std::string mouse_device = "";
    FILE* pipe = popen(mouse_cmd.c_str(), "r");
    while (fgets(buffer_cmd, sizeof buffer_cmd, pipe) != NULL) {
        mouse_device += buffer_cmd;
    }
    pclose(pipe);


    if (mouse_device == "") {
      printf("valid mouse not found, plug it in and try again\n");
      exit(1);
    }
    else {
      mouse_device = "/dev/input/" + mouse_device;
      mouse_device.erase(std::remove(mouse_device.begin(), mouse_device.end(), '\n'), mouse_device.cend());
      mouse_device.erase(std::remove(mouse_device.begin(), mouse_device.end(), ' '), mouse_device.cend());
      printf("using device: %s\n", mouse_device.c_str());
    }

    mouse_t mouse = mouse_t(const_cast<char*>(mouse_device.c_str()));
    global_mouse = &mouse;
    stopwatch_t stopwatch = stopwatch_t();
    buffer_t* buffer = A;
    buffer_t* otherBuffer = B;
    buffer_t* temp;
    mouse.getData();
    stopwatch.click();
    while(*running){
        mouse.getData();
        stopwatch.click();
        entry_t entry = {mouse.x, mouse.y, stopwatch.refreshTime()};
        if(buffer->push(entry)){
            printf("[COLLECTOR]: Buffer full, switching buffers...\n");
            temp = buffer;
            buffer = otherBuffer;
            otherBuffer = temp;
        }
    }
    printf("[COLLECTOR]: EXIT\n");
}

void interrupt_handler(int signum){
    *running = false;
    printf("Interrupt signal (%d) received.\n", signum);
    A->cv->notify_all();
    B->cv->notify_all();
}

void archiver_thread(){
    // Get current date and time as string
    time_t now = time(0);
    tm* ltm = localtime(&now);
    char* date = new char[180];
    strftime(date, 180, "%Y-%m-%d_%H-%M-%S", ltm);
    char* filename = new char[180];

    if (cli_filename == "" || cli_filename == "NULL") {
      sprintf(filename, "./data/data_%s.csv", date);
    }
    else {
      sprintf(filename, cli_filename.c_str(), date);
    }

    file_handler file = file_handler(filename);
    buffer_t* buffer = A;
    buffer_t* otherBuffer = B;
    buffer_t* temp;
    char* str = new char[BUFFER_SIZE * 300];

    file.write("offset, x, y\n");

    while(*running){
        printf("[ARCHIVER]: Waiting for buffer to fill...\n");
        buffer->wait();
        if(!*running){
            break;
        }
        printf("[ARCHIVER]: Buffer full, writing to file...\n");
        buffer->stringify(str);
        file.write(str);
        buffer->reset();
        temp = buffer;
        buffer = otherBuffer;
        otherBuffer = temp;
    }
    printf("[ARCHIVER]: EXITING: DUMP REMAINING BUFFER\n");
    buffer->stringify(str);
    file.write(str);
    printf("[ARCHIVER]: EXIT\n");
    delete[] str;
}

int main(int argc, char** argv) {


    if (argc > 3) {
      printf("too many arguments! (%s)", argc);
    }

    if (argc == 3) {

      int m_type_val = std::stoi(argv[2]);
      //printf("%i: mtype\n", m_type_val);
      m_type = static_cast<MOUSE_TYPE>(m_type_val);

    }
    else {
      std::cout << "[0]: Razer (8KHz)" << std::endl;
      std::cout << "[1]: G502  (1KHz)" << std::endl;
      std::cout << "choose a mouse from the above selection: ";

      int m_sel;
      std::cin >> m_sel;

      if (m_sel < 0 || m_sel > static_cast<int>(MOUSE_TYPE::NONE)) {
        std::cout << "invalid mouse selection, exiting" << std::endl;
        exit(1);
      }

      m_type = static_cast<MOUSE_TYPE>(m_sel);
    }

    auto g = poll_map.find(m_type);
    //std::cout << static_cast<int>(g->first) << " " << g->second << std::endl; 
    SAMPLE_RATE = g->second;


    switch (m_type) {
      using enum MOUSE_TYPE;
      case RAZER8KHZ:
        mouse_cmd = "cat /proc/bus/input/devices | grep -A5 -ne \"\\\"Razer Razer Viper 8KHz\\\"\" | sed '/Handlers=.* mouse[0-9]/!d' | sed -E 's/(.*)(mouse.*)/\\2/g'";
        break;

      case G502:
        mouse_cmd = "cat /proc/bus/input/devices | grep -A5 -ne \"\\\"Logitech G502 HERO Gaming Mouse\\\"\" | sed '/Handlers=.* mouse[0-9]/!d' | sed -E 's/(.*)(mouse.*)/\\2/g'";
        break;
      default:
        std::cerr << "bad mouse" << std::endl;
        exit(1);
    }
      

    if (argc >= 2) {
      cli_filename = argv[1];
      if (cli_filename != "" && cli_filename != "NULL") {
        printf("using %s as CSV output path\n", cli_filename.c_str());
      }
    }
    signal(SIGINT, interrupt_handler);

    std::thread collector(collector_thread);
    std::thread archiver(archiver_thread);

    printf("Executing...\n");
    archiver.join();
   
    global_mouse->endData();

    collector.join();

    printf("Exiting...\n");
    return 0; 
}
