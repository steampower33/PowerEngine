// library_impl.cpp ���� ����

// TinyGLTF�� ������ �����մϴ�.
#define TINYGLTF_IMPLEMENTATION

// TinyGLTF�� �̹����� �ε��� �� stb_image.h�� �����մϴ�.
// �̹����� �ε��ϴ� �Լ�(stbi_load_from_memory ��)�� �����ϱ� ���� �ʿ��մϴ�.
#define STB_IMAGE_IMPLEMENTATION 

// TinyGLTF�� �̹��� ���� ���⸦ ���� �ʿ�� �մϴ�.
#define STB_IMAGE_WRITE_IMPLEMENTATION 

// TinyGLTF�� �̹��� ���� ������ �б� ���� �ʿ��� �� �ֽ��ϴ�.
#define STB_IMAGE_STATIC

#include <tiny_gltf.h>
// stb_image.h�� ���� ��Ŭ����� �ʿ�� ���� ������, tinygltf.h�� ���������� ó���մϴ�.
// ���� tinygltf.h�� ������� �ʴٸ� �Ʒ��� �߰��ؾ� �մϴ�.
// #include <stb_image.h>