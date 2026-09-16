#include <gfx/radiance_rate/textured_obj_scene.hpp>

#include <gfx/research/application.hpp>
#include <gfx/research/assets.hpp>
#include <gfx/research/framebuffer.hpp>
#include <gfx/research/fullscreen_triangle.hpp>
#include <gfx/research/gpu_timer.hpp>
#include <gfx/research/orbit_camera.hpp>
#include <gfx/research/shader.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glad/gl.h>
#include <imgui.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace {

enum class Method : int {
    FullRate = 0,
    Fixed2x2 = 1,
    Fixed4x4 = 2,
    Heuristic = 3,
    RadianceBudget = 4
};

constexpr std::array<const char*, 5> methodLabels = {
    "Full-rate reference",
    "Fixed 2x2",
    "Fixed 4x4",
    "Screen-space heuristic",
    "GGX radiance-error budget"
};

struct LaunchOptions {
    bool quick = false;
    bool full = false;
    std::filesystem::path output = "results/publication";
};

struct Metrics {
    double rmse = 0.0;
    double relativeRmse = 0.0;
    double gradientRmse = 0.0;
    double maxAbs = 0.0;
    double violationRate = 0.0;
    double rate1 = 0.0;
    double rate2 = 0.0;
    double rate4 = 0.0;
};

struct Case {
    Method method = Method::RadianceBudget;
    int width = 1280;
    int height = 720;
    int lights = 64;
    int cameraPreset = 0;
    float roughnessScale = 1.0f;
    float budget = 0.025f;
};

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t m = values.size() / 2;
    return values.size() % 2 == 0 ? 0.5 * (values[m - 1] + values[m]) : values[m];
}

std::vector<float> readRgba(unsigned int texture, int width, int height) {
    std::vector<float> data(static_cast<std::size_t>(width) * height * 4);
    glGetTextureImage(texture, 0, GL_RGBA, GL_FLOAT, static_cast<GLsizei>(data.size() * sizeof(float)), data.data());
    return data;
}

std::vector<float> readR(unsigned int texture, int width, int height) {
    std::vector<float> data(static_cast<std::size_t>(width) * height);
    glGetTextureImage(texture, 0, GL_RED, GL_FLOAT, static_cast<GLsizei>(data.size() * sizeof(float)), data.data());
    return data;
}

double luminance(const std::vector<float>& a, std::size_t k) {
    return 0.2126 * a[k] + 0.7152 * a[k + 1] + 0.0722 * a[k + 2];
}

Metrics calculateMetrics(
    const std::vector<float>& image,
    const std::vector<float>& reference,
    const std::vector<float>& rates,
    int width,
    int height,
    float budget
) {
    Metrics out{};
    if (image.size() != reference.size() || image.empty()) return out;
    long double sq = 0.0;
    long double refSq = 0.0;
    long double gradSq = 0.0;
    std::size_t n = 0;
    std::size_t violations = 0;
    for (std::size_t k = 0; k + 3 < image.size(); k += 4) {
        for (int c = 0; c < 3; ++c) {
            const double d = image[k + c] - reference[k + c];
            sq += d * d;
            refSq += static_cast<double>(reference[k + c]) * reference[k + c];
            out.maxAbs = std::max(out.maxAbs, std::abs(d));
            ++n;
        }
        if (std::abs(luminance(image, k) - luminance(reference, k)) > budget) ++violations;
    }
    out.rmse = std::sqrt(static_cast<double>(sq / std::max<std::size_t>(n, 1)));
    out.relativeRmse = std::sqrt(static_cast<double>(sq / std::max<long double>(refSq, 1e-20L)));
    out.violationRate = static_cast<double>(violations) / static_cast<double>(image.size() / 4);

    auto lum = [&](const std::vector<float>& a, int x, int y) {
        return luminance(a, (static_cast<std::size_t>(y) * width + x) * 4);
    };
    std::size_t gn = 0;
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            const double dx = (lum(image, x + 1, y) - lum(image, x, y))
                - (lum(reference, x + 1, y) - lum(reference, x, y));
            const double dy = (lum(image, x, y + 1) - lum(image, x, y))
                - (lum(reference, x, y + 1) - lum(reference, x, y));
            gradSq += dx * dx + dy * dy;
            gn += 2;
        }
    }
    out.gradientRmse = std::sqrt(static_cast<double>(gradSq / std::max<std::size_t>(gn, 1)));

    if (rates.size() == static_cast<std::size_t>(width) * height) {
        for (float r : rates) {
            if (r < 1.5f) ++out.rate1;
            else if (r < 3.0f) ++out.rate2;
            else ++out.rate4;
        }
        const double total = static_cast<double>(rates.size());
        out.rate1 /= total;
        out.rate2 /= total;
        out.rate4 /= total;
    }
    return out;
}

bool savePfm(const std::filesystem::path& path, unsigned int texture, int width, int height) {
    const auto rgba = readRgba(texture, width, height);
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "PF\n" << width << ' ' << height << "\n-1.0\n";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t k = (static_cast<std::size_t>(y) * width + x) * 4;
            out.write(reinterpret_cast<const char*>(&rgba[k]), sizeof(float) * 3);
        }
    }
    return out.good();
}

glm::mat4 fittedSceneTransform(
    const gfx::radiance_rate::TexturedObjScene& scene,
    const glm::vec3& groundPosition,
    float targetSize
) {
    const glm::vec3 minimum = scene.bounds_min();
    const glm::vec3 maximum = scene.bounds_max();
    const glm::vec3 extent = maximum - minimum;
    const float largest = std::max({extent.x, extent.y, extent.z, 1e-5f});
    const float scale = targetSize / largest;
    const glm::vec3 anchor(
        0.5f * (minimum.x + maximum.x),
        minimum.y,
        0.5f * (minimum.z + maximum.z)
    );
    glm::mat4 m(1.0f);
    m = glm::translate(m, groundPosition);
    m = glm::scale(m, glm::vec3(scale));
    m = glm::translate(m, -anchor);
    return m;
}

std::string methodId(Method m) {
    if (m == Method::FullRate) return "full-rate";
    if (m == Method::Fixed2x2) return "fixed-2x2";
    if (m == Method::Fixed4x4) return "fixed-4x4";
    if (m == Method::Heuristic) return "heuristic";
    return "radiance-budget";
}

class RadianceRateResearch final : public gfx::research::Application {
public:
    explicit RadianceRateResearch(LaunchOptions options)
        : Application({.title = "gfx-research-radiance-rate", .width = 1600, .height = 900, .vsync = false}),
          options_(std::move(options)) {}

private:
    bool on_init() override {
        root_ = GFX_RESEARCH_SOURCE_DIR;
        const gfx::research::ShaderFile gbufferFiles[] = {
            {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
            {gfx::research::ShaderStage::Fragment, root_ / "shaders/gbuffer.frag"}
        };
        const gfx::research::ShaderFile lightingFiles[] = {
            {gfx::research::ShaderStage::Compute, root_ / "shaders/adaptive_lighting.comp"}
        };
        const gfx::research::ShaderFile displayFiles[] = {
            {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
            {gfx::research::ShaderStage::Fragment, root_ / "shaders/display.frag"}
        };
        if (!gbufferShader_.load(gbufferFiles) || !lightingShader_.load(lightingFiles) || !displayShader_.load(displayFiles)) {
            return false;
        }

        if (!scene_.load(gfx::research::model_path(gfx::research::ModelAsset::Sponza))) {
            std::cerr << "Failed to load Sponza from gfx-research-base assets\n";
            return false;
        }

        camera_.set_target({0.0f, 1.35f, 0.0f});
        camera_.set_distance(4.05f);
        camera_.set_yaw(1.54f);
        camera_.set_pitch(-0.06f);
        camera_.set_fov(54.0f);
        camera_.set_clip(0.03f, 250.0f);

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);

        if (!createTargets(config().width, config().height)) return false;
        if (options_.quick || options_.full) startBenchmark(options_.full);
        return true;
    }

    void on_resize(int width, int height) override {
        if (!benchmarkRunning_) createTargets(width, height);
    }

    void on_frame(const gfx::research::FrameInfo& frame) override {
        if (benchmarkRunning_) benchmarkFrame();
        else {
            renderGbuffer(frame.width, frame.height);
            renderLighting(method_, result_, frame.width, frame.height, true);
        }

        gfx::research::Framebuffer::bind_default();
        glViewport(0, 0, frame.width, frame.height);
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.012f, 0.014f, 0.020f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        displayShader_.bind();
        displayShader_.set_int("uHdr", 0);
        displayShader_.set_int("uRate", 1);
        displayShader_.set_int("uShowRate", showRate_ ? 1 : 0);
        displayShader_.set_float("uExposure", displayExposure_);
        glBindTextureUnit(0, result_.color(0));
        glBindTextureUnit(1, result_.color(1));
        fullscreen_.draw();
        glEnable(GL_DEPTH_TEST);
    }

    void on_gui(const gfx::research::FrameInfo&) override {
        if (benchmarkRunning_) {
            ImGui::Begin("Publication benchmark");
            ImGui::Text("Case %zu / %zu", benchmarkIndex_ + 1, cases_.size());
            ImGui::Text("Frame %d / %d", frameInCase_ + 1, warmupFrames_ + sampleFrames_);
            ImGui::ProgressBar(
                cases_.empty() ? 0.0f : static_cast<float>(benchmarkIndex_) / cases_.size(),
                ImVec2(-1.0f, 0.0f)
            );
            ImGui::End();
            return;
        }

        ImGui::Begin("Radiance-error shading-rate research");
        int method = static_cast<int>(method_);
        if (ImGui::Combo("Method", &method, methodLabels.data(), static_cast<int>(methodLabels.size()))) {
            method_ = static_cast<Method>(method);
        }
        ImGui::SliderInt("Point lights", &lightCount_, 1, 256);
        ImGui::SliderFloat("Radiance budget", &errorBudget_, 0.001f, 0.12f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Roughness scale", &roughnessScale_, 0.25f, 2.0f);
        ImGui::Checkbox("Show shading rate", &showRate_);
        ImGui::SliderFloat("Display exposure", &displayExposure_, 0.2f, 2.0f);
        ImGui::Separator();
        ImGui::SliderFloat("Camera distance", &camera_.distance(), 1.5f, 7.0f);
        ImGui::SliderAngle("Camera yaw", &camera_.yaw(), -180.0f, 180.0f);
        ImGui::SliderAngle("Camera pitch", &camera_.pitch(), -20.0f, 55.0f);
        ImGui::Text("Lighting GPU %.3f ms", gpuTimer_.has_result() ? gpuTimer_.milliseconds() : 0.0);
        if (ImGui::Button("Run quick suite")) startBenchmark(false);
        ImGui::SameLine();
        if (ImGui::Button("Run full suite")) startBenchmark(true);
        ImGui::End();
    }

    bool createTargets(int width, int height) {
        if (width <= 0 || height <= 0) return false;
        const gfx::research::ColorAttachmentDesc gbuf[] = {
            {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
            {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
            {GL_RGBA16F, GL_NEAREST, GL_NEAREST}
        };
        const gfx::research::ColorAttachmentDesc result[] = {
            {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
            {GL_R16F, GL_NEAREST, GL_NEAREST}
        };
        renderWidth_ = width;
        renderHeight_ = height;
        return gbuffer_.create(width, height, gbuf, true, GL_DEPTH_COMPONENT32F)
            && result_.create(width, height, result, false)
            && reference_.create(width, height, result, false);
    }

    void applyCameraPreset(int preset) {
        if (preset == 0) {
            camera_.set_target({0.0f, 1.35f, 0.0f});
            camera_.set_distance(4.05f);
            camera_.set_yaw(1.54f);
            camera_.set_pitch(-0.06f);
        } else if (preset == 1) {
            camera_.set_target({0.0f, 1.10f, 0.0f});
            camera_.set_distance(3.1f);
            camera_.set_yaw(0.15f);
            camera_.set_pitch(0.08f);
        } else {
            camera_.set_target({0.0f, 1.65f, 0.0f});
            camera_.set_distance(5.3f);
            camera_.set_yaw(-1.05f);
            camera_.set_pitch(0.18f);
        }
    }

    void renderGbuffer(int width, int height) {
        gbuffer_.bind();
        glViewport(0, 0, width, height);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        const float zero[] = {0, 0, 0, 0};
        glClearNamedFramebufferfv(gbuffer_.id(), GL_COLOR, 0, zero);
        glClearNamedFramebufferfv(gbuffer_.id(), GL_COLOR, 1, zero);
        glClearNamedFramebufferfv(gbuffer_.id(), GL_COLOR, 2, zero);
        const float clearDepth = 1.0f;
        glClearNamedFramebufferfv(gbuffer_.id(), GL_DEPTH, 0, &clearDepth);

        const glm::mat4 model = fittedSceneTransform(scene_, {0.0f, 0.0f, 0.0f}, 14.0f);
        const glm::mat4 vp = camera_.view_projection(static_cast<float>(width) / static_cast<float>(height));
        gbufferShader_.bind();
        gbufferShader_.set_mat4("uModel", model);
        gbufferShader_.set_mat4("uMvp", vp * model);
        gbufferShader_.set_float("uRoughnessScale", roughnessScale_);
        scene_.draw(gbufferShader_);
    }

    void renderLighting(Method method, gfx::research::Framebuffer& target, int width, int height, bool timed) {
        if (timed) gpuTimer_.begin();
        lightingShader_.bind();
        lightingShader_.set_int("uAlbedoRoughness", 0);
        lightingShader_.set_int("uNormalValid", 1);
        lightingShader_.set_int("uPositionValid", 2);
        lightingShader_.set_int("uMode", static_cast<int>(method));
        lightingShader_.set_int("uLightCount", lightCount_);
        lightingShader_.set_float("uErrorBudget", errorBudget_);
        lightingShader_.set_vec3("uEye", camera_.position());
        lightingShader_.set_uint("uFrameSeed", 0u);
        glBindTextureUnit(0, gbuffer_.color(0));
        glBindTextureUnit(1, gbuffer_.color(1));
        glBindTextureUnit(2, gbuffer_.color(2));
        glBindImageTexture(0, target.color(0), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glBindImageTexture(1, target.color(1), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R16F);
        glDispatchCompute(
            static_cast<GLuint>((width + 7) / 8),
            static_cast<GLuint>((height + 7) / 8),
            1
        );
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        if (timed) gpuTimer_.end();
    }

    void applyCase(const Case& c) {
        method_ = c.method;
        lightCount_ = c.lights;
        roughnessScale_ = c.roughnessScale;
        errorBudget_ = c.budget;
        applyCameraPreset(c.cameraPreset);
        createTargets(c.width, c.height);
        renderGbuffer(c.width, c.height);
        renderLighting(Method::FullRate, reference_, c.width, c.height, false);
        referenceImage_ = readRgba(reference_.color(0), c.width, c.height);
        timings_.clear();
        metrics_ = {};
        metricsCaptured_ = false;
        frameInCase_ = 0;
    }

    void startBenchmark(bool full) {
        cases_.clear();
        const std::array<Method, 4> methods = {
            Method::Fixed2x2, Method::Fixed4x4, Method::Heuristic, Method::RadianceBudget
        };
        if (full) {
            const std::array<std::pair<int, int>, 3> resolutions = {{{1280, 720}, {1920, 1080}, {2560, 1440}}};
            const std::array<int, 3> lights = {16, 64, 256};
            const std::array<float, 3> roughness = {0.55f, 1.0f, 1.6f};
            const std::array<float, 3> budgets = {0.010f, 0.025f, 0.050f};
            for (Method m : methods) for (auto [w, h] : resolutions)
                for (int l : lights) for (int camera = 0; camera < 3; ++camera)
                    for (float r : roughness) for (float b : budgets)
                        cases_.push_back({m, w, h, l, camera, r, b});
            warmupFrames_ = 30;
            sampleFrames_ = 120;
        } else {
            for (Method m : methods) {
                cases_.push_back({m, 1280, 720, 16, 0, 1.0f, 0.025f});
                cases_.push_back({m, 1280, 720, 64, 1, 1.0f, 0.025f});
                cases_.push_back({m, 1280, 720, 256, 2, 0.55f, 0.025f});
                cases_.push_back({m, 1280, 720, 64, 0, 1.6f, 0.050f});
            }
            warmupFrames_ = 10;
            sampleFrames_ = 30;
        }

        std::filesystem::create_directories(options_.output);
        csv_.open(options_.output / "raw.csv", std::ios::trunc);
        csv_ << "method,width,height,lights,camera,roughness_scale,error_budget,gpu_median_ms,"
                "rmse,relative_rmse,gradient_rmse,max_abs,violation_rate,rate1,rate2,rate4\n";
        benchmarkIndex_ = 0;
        benchmarkRunning_ = !cases_.empty();
        if (benchmarkRunning_) applyCase(cases_[0]);
    }

    void benchmarkFrame() {
        if (benchmarkIndex_ >= cases_.size()) {
            benchmarkRunning_ = false;
            if (csv_.is_open()) csv_.close();
            close();
            return;
        }

        const Case& c = cases_[benchmarkIndex_];
        renderGbuffer(c.width, c.height);
        renderLighting(c.method, result_, c.width, c.height, true);
        if (frameInCase_ >= warmupFrames_ && gpuTimer_.has_result()) timings_.push_back(gpuTimer_.milliseconds());

        if (!metricsCaptured_ && frameInCase_ == warmupFrames_) {
            const auto image = readRgba(result_.color(0), c.width, c.height);
            const auto rates = readR(result_.color(1), c.width, c.height);
            metrics_ = calculateMetrics(image, referenceImage_, rates, c.width, c.height, c.budget);
            metricsCaptured_ = true;
            if (c.method == Method::RadianceBudget && c.lights == 256 && c.cameraPreset == 2) {
                savePfm(options_.output / "captures" / "stress-method.pfm", result_.color(0), c.width, c.height);
                savePfm(options_.output / "captures" / "stress-reference.pfm", reference_.color(0), c.width, c.height);
            }
        }

        ++frameInCase_;
        if (frameInCase_ < warmupFrames_ + sampleFrames_) return;

        csv_ << methodId(c.method) << ',' << c.width << ',' << c.height << ',' << c.lights << ','
             << c.cameraPreset << ',' << c.roughnessScale << ',' << c.budget << ','
             << median(timings_) << ',' << metrics_.rmse << ',' << metrics_.relativeRmse << ','
             << metrics_.gradientRmse << ',' << metrics_.maxAbs << ',' << metrics_.violationRate << ','
             << metrics_.rate1 << ',' << metrics_.rate2 << ',' << metrics_.rate4 << '\n';
        csv_.flush();

        ++benchmarkIndex_;
        if (benchmarkIndex_ < cases_.size()) applyCase(cases_[benchmarkIndex_]);
        else {
            benchmarkRunning_ = false;
            csv_.close();
            close();
        }
    }

    LaunchOptions options_;
    std::filesystem::path root_;
    gfx::radiance_rate::TexturedObjScene scene_;
    gfx::research::Shader gbufferShader_;
    gfx::research::Shader lightingShader_;
    gfx::research::Shader displayShader_;
    gfx::research::Framebuffer gbuffer_;
    gfx::research::Framebuffer result_;
    gfx::research::Framebuffer reference_;
    gfx::research::FullscreenTriangle fullscreen_;
    gfx::research::GpuTimer gpuTimer_;
    gfx::research::OrbitCamera camera_;

    Method method_ = Method::RadianceBudget;
    int lightCount_ = 64;
    float roughnessScale_ = 1.0f;
    float errorBudget_ = 0.025f;
    float displayExposure_ = 1.0f;
    bool showRate_ = false;
    int renderWidth_ = 1600;
    int renderHeight_ = 900;

    bool benchmarkRunning_ = false;
    std::vector<Case> cases_;
    std::size_t benchmarkIndex_ = 0;
    int frameInCase_ = 0;
    int warmupFrames_ = 10;
    int sampleFrames_ = 30;
    std::vector<double> timings_;
    std::vector<float> referenceImage_;
    Metrics metrics_{};
    bool metricsCaptured_ = false;
    std::ofstream csv_;
};

LaunchOptions parseOptions(int argc, char** argv) {
    LaunchOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--publication-quick") options.quick = true;
        else if (arg == "--publication-suite") options.full = true;
        else if (arg.starts_with("--output=")) options.output = std::string(arg.substr(9));
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    return RadianceRateResearch{parseOptions(argc, argv)}.run();
}
