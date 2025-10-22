// library_impl.cpp 파일 내용

// TinyGLTF의 구현을 생성합니다.
#define TINYGLTF_IMPLEMENTATION

// TinyGLTF는 이미지를 로드할 때 stb_image.h에 의존합니다.
// 이미지를 로드하는 함수(stbi_load_from_memory 등)를 구현하기 위해 필요합니다.
#define STB_IMAGE_IMPLEMENTATION 

// TinyGLTF가 이미지 파일 쓰기를 위해 필요로 합니다.
#define STB_IMAGE_WRITE_IMPLEMENTATION 

// TinyGLTF가 이미지 파일 정보를 읽기 위해 필요할 수 있습니다.
#define STB_IMAGE_STATIC

#include <tiny_gltf.h>
// stb_image.h를 직접 인클루드할 필요는 보통 없지만, tinygltf.h가 내부적으로 처리합니다.
// 만약 tinygltf.h가 충분하지 않다면 아래도 추가해야 합니다.
// #include <stb_image.h>