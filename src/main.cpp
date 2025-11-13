#include "window.h"

int main() {
    try {
        Window win;
        win.mainloop();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}