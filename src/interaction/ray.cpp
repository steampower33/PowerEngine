#include "model.h"
#include <limits>

#include "ray.h"

Ray::Ray(const glm::vec3& origin, const glm::vec3& direction)
    : origin(origin), direction(glm::normalize(direction)) 
{
}

bool Ray::Intersects(const Model& model, float& dist) const
{
    glm::vec3 oc = origin - model.position_;

    const float a = 1.0f;

    const float b = 2.0f * glm::dot(oc, direction);

    const float c = glm::dot(oc, oc) - model.radius_ * model.radius_;

    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrt_discriminant = sqrt(discriminant);
    const float t0 = (-b - sqrt_discriminant) / (2.0f * a);
    const float t1 = (-b + sqrt_discriminant) / (2.0f * a);

    float closest_t = std::numeric_limits<float>::max();
    bool hit = false;

    if (t0 > 0.0001f) {
        closest_t = t0;
        hit = true;
    }

    if (t1 > 0.0001f && t1 < closest_t) {
        closest_t = t1;
        hit = true;
    }

    if (hit) {
        dist = closest_t;
        return true;
    }

    return false;
}