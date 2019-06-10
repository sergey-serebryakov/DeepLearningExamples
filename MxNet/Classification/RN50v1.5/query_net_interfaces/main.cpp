#include <sys/types.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <ifaddrs.h>

#include <iostream>
#include <string>
using namespace std;



int main(int argc, char **argv) {
    struct ifaddrs * ifAddrStruct = nullptr;
    struct ifaddrs * ifa = nullptr;
    
    getifaddrs(&ifAddrStruct);
    unsigned int num_found = 0;
    for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
        if (nullptr == ifa->ifa_addr) continue;

        if (AF_INET == ifa->ifa_addr->sa_family && 0 == (ifa->ifa_flags & IFF_LOOPBACK)) {
            char address_buffer[INET_ADDRSTRLEN];
            void* sin_addr_ptr = &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr;
            inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);

            std::cout << "Interface='" << std::string(ifa->ifa_name)
                      << "' with IP address='" << std::string(address_buffer)
                      << "'" << std::endl;
            num_found += 1;
        }
    }
    if (nullptr != ifAddrStruct) freeifaddrs(ifAddrStruct);
  
    if (num_found == 0) {
        std::cout << "No network interfaces found" << std::endl;
    }
    return 0;
}
