#include "camera.h"

glm::vec3 Camera::Front() const {
    float cy = cos(glm::radians(yaw));
    float sy = sin(glm::radians(yaw));
    float cp = cos(glm::radians(pitch));
    float sp = sin(glm::radians(pitch));

    //std::cout << position.x << " " << position.y << " " << position.z << std::endl;
    //std::cout << yaw << " " << pitch << " " << std::endl;
    return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
}
glm::vec3 Camera::Right() const {
    return glm::normalize(glm::cross(Front(), glm::vec3(0, 1, 0)));
}
glm::vec3 Camera::Up() const {
    return glm::normalize(glm::cross(Right(), Front()));
}

glm::mat4 Camera::View() const {
    return glm::lookAt(position, position + Front(), glm::vec3(0, 1, 0));
}
glm::mat4 Camera::Proj(float width, float height) const {
    glm::mat4 p = glm::perspective(glm::radians(fov), width / height, 0.1f, 20.0f);
    //p[1][1] *= -1;
    return p;
}