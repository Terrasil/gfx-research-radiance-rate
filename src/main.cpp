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
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
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
    RadianceBudget = 4,
    FeatureBudget = 5,
    ParallelFeature = 6,
    DirectFullRate = 7
};

constexpr std::array<const char*, 8> methodLabels = {
    "Full-rate reference",
    "Fixed 2x2",
    "Fixed 4x4",
    "Screen-space heuristic",
    "GGX radiance-error budget",
    "Serial feature gate",
    "Parallel feature gate",
    "Direct full rate"
};

struct LaunchOptions {
    bool smoke = false;
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
    double predictorRmse = 0.0; // per 8x8 tile, luminance max error versus proxy
    double falseSafeRate = 0.0; // fraction of proxy-selected coarse tiles that violate the budget
    double maxUnderestimate = 0.0;
    std::size_t predictorTiles = 0;
};

struct Case {
    Method method = Method::RadianceBudget;
    int width = 1280;
    int height = 720;
    int lights = 64;
    int cameraPreset = 0;
    float roughnessScale = 1.0f;
    float budget = 0.025f;
    int block = 0;
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
            if (!std::isfinite(image[k+c]) || !std::isfinite(reference[k+c]))
                throw std::runtime_error("A lighting pass produced a nonfinite RGB sample");
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
    // The predictor is carried in the otherwise unused HDR alpha channel,
    // at each pixel of the tile. Record realized maximum luminance error
    // instead of comparing the proxy with an unrelated full-frame RMSE.
    for (int ty=0; ty<height; ty+=8) for (int tx=0; tx<width; tx+=8) {
        const auto first=(static_cast<std::size_t>(ty)*width+tx)*4;
        const double predicted=image[first+3];
        if (!std::isfinite(predicted) || predicted<0.0) continue;
        double actual=0.0;
        for(int y=ty; y<std::min(ty+8,height); ++y)
            for(int x=tx; x<std::min(tx+8,width); ++x) {
                const auto k=(static_cast<std::size_t>(y)*width+x)*4;
                actual=std::max(actual,std::abs(luminance(image,k)-luminance(reference,k)));
            }
        out.predictorRmse+=(predicted-actual)*(predicted-actual);
        out.maxUnderestimate=std::max(out.maxUnderestimate,actual-predicted);
        if(actual>budget) out.falseSafeRate+=1.0;
        ++out.predictorTiles;
    }
    if(out.predictorTiles) {
        out.predictorRmse=std::sqrt(out.predictorRmse/double(out.predictorTiles));
        out.falseSafeRate/=double(out.predictorTiles);
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
    if (m == Method::FeatureBudget) return "feature-budget";
    if (m == Method::ParallelFeature) return "parallel-feature";
    if (m == Method::DirectFullRate) return "full-rate-direct";
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
        camera_.set_distance(4.110f);
        camera_.set_yaw(1.570796327f); // 90 degrees
        camera_.set_pitch(0.0f);       // 0 degrees
        camera_.set_fov(54.0f);
        camera_.set_clip(0.03f, 250.0f);

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);

        if (!createTargets(config().width, config().height)) return false;
        if (runsBenchmarkFromCli()) startBenchmark(options_.full, options_.smoke);
        return true;
    }

    void on_resize(int width, int height) override {
        if (!benchmarkRunning_) createTargets(width, height);
    }

    void on_frame(const gfx::research::FrameInfo& frame) override {
        if (benchmarkRunning_) {
            try {
                benchmarkFrame();
            } catch (const std::exception& error) {
                if (runsBenchmarkFromCli()) throw;
                finishBenchmark(false);
                benchmarkMessage_ = error.what();
                std::cerr << benchmarkMessage_ << '\n';
            }
        } else {
            if ((renderWidth_ != frame.width || renderHeight_ != frame.height)
                && !createTargets(frame.width, frame.height))
                throw std::runtime_error("Could not restore viewer targets");
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
        const auto* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({viewport->WorkPos.x+12,viewport->WorkPos.y+12},ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({448,std::min(800.0f,std::max(320.0f,viewport->WorkSize.y-24))},ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSizeConstraints({380,320},{650,std::max(320.0f,viewport->WorkSize.y-24)});
        if (!ImGui::Begin("Radiance Rate Research")) { ImGui::End(); return; }
        ImGui::Text("Lighting GPU %.3f ms | FPS %.0f",gpuTimer_.has_result()?gpuTimer_.milliseconds():0.0,ImGui::GetIO().Framerate);
        if (benchmarkRunning_) {
            ImGui::SeparatorText("Publication benchmark");
            ImGui::Text("Case %zu / %zu", benchmarkIndex_+1, cases_.size());
            ImGui::Text("Frame %d / %d", frameInCase_+1,warmupFrames_+sampleFrames_);
            ImGui::ProgressBar(cases_.empty()?0.0f:static_cast<float>(benchmarkIndex_)/cases_.size(),{-1,0});
            ImGui::TextWrapped("%s", benchmarkOutput_.string().c_str());
            if (ImGui::Button("Stop benchmark", {-1, 0})) finishBenchmark(false);
            ImGui::End(); return;
        }
        if (ImGui::BeginTabBar("Sections")) {
            if (ImGui::BeginTabItem("Rate")) {
                int method = static_cast<int>(method_);
                if (ImGui::Combo("Method",&method,methodLabels.data(),static_cast<int>(methodLabels.size()))) method_=static_cast<Method>(method);
                ImGui::SliderInt("Point lights",&lightCount_,1,256);
                ImGui::SliderFloat("Radiance budget",&errorBudget_,0.001f,0.12f,"%.4f",ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Roughness scale",&roughnessScale_,0.25f,2.0f);
                ImGui::TextWrapped("Feature budget is a separate low-cost ablation. Historic method IDs and their defaults are unchanged.");
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Scene")) {
                ImGui::Checkbox("Show shading rate",&showRate_);
                ImGui::SliderFloat("Display exposure",&displayExposure_,0.2f,2.0f);
                ImGui::TextDisabled("Timing and HDR metrics use offscreen benchmark targets.");
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Camera")) {
                ImGui::SliderFloat("Camera distance",&camera_.distance(),1.5f,7.0f);
                ImGui::SliderAngle("Camera yaw",&camera_.yaw(),-180.0f,180.0f);
                ImGui::SliderAngle("Camera pitch",&camera_.pitch(),-20.0f,55.0f);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Tests")) {
                ImGui::SeparatorText("Publication benchmarks");
                const auto launch = [&](const char* label, bool full, bool smoke = false) {
                    if (!ImGui::Button(label, {-1, 0})) return;
                    try {
                        startBenchmark(full, smoke);
                    } catch (const std::exception& error) {
                        finishBenchmark(false);
                        benchmarkMessage_ = error.what();
                    }
                    // on_gui is called after the scene. Leave the benchmark G-buffer intact
                    // for the next frame instead of drawing the panel into its attachments.
                    gfx::research::Framebuffer::bind_default();
                };
                launch("Run smoke test", false, true);
                launch("Run quick suite", false);
                launch("Run full suite", true);
                if (!benchmarkMessage_.empty()) ImGui::TextWrapped("%s", benchmarkMessage_.c_str());
                if (!benchmarkOutput_.empty()) ImGui::TextWrapped("Output: %s", benchmarkOutput_.string().c_str());
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
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
            camera_.set_distance(4.110f);
            camera_.set_yaw(1.570796327f); // 90 degrees
            camera_.set_pitch(0.0f);       // 0 degrees
        } else if (preset == 1) {
            camera_.set_target({0.0f, 1.35f, 0.0f});
            camera_.set_distance(4.630f);
            camera_.set_yaw(1.710422667f); // 98 degrees
            camera_.set_pitch(0.034906585f); // 2 degrees
        } else {
            camera_.set_target({0.0f, 1.35f, 0.0f});
            camera_.set_distance(2.050f);
            camera_.set_yaw(1.169370599f); // 67 degrees
            camera_.set_pitch(0.593411946f); // 34 degrees
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

    void renderLighting(Method method, gfx::research::Framebuffer& target, int width, int height, bool timed,
                        GLuint elapsedQuery = 0) {
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
        if (elapsedQuery) glBeginQuery(GL_TIME_ELAPSED,elapsedQuery);
        glDispatchCompute(
            static_cast<GLuint>((width + 7) / 8),
            static_cast<GLuint>((height + 7) / 8),
            1
        );
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        if (elapsedQuery) glEndQuery(GL_TIME_ELAPSED);
        if (timed) gpuTimer_.end();
    }

    void applyCase(const Case& c) {
        method_ = c.method;
        lightCount_ = c.lights;
        roughnessScale_ = c.roughnessScale;
        errorBudget_ = c.budget;
        applyCameraPreset(c.cameraPreset);
        if (!createTargets(c.width,c.height)) throw std::runtime_error("Could not create benchmark targets");
        renderGbuffer(c.width, c.height);
        renderLighting(Method::FullRate, reference_, c.width, c.height, false);
        referenceImage_ = readRgba(reference_.color(0), c.width, c.height);
        timings_.clear();
        metrics_ = {};
        metricsCaptured_ = false;
        frameInCase_ = 0;
    }

    bool runsBenchmarkFromCli() const { return options_.smoke || options_.quick || options_.full; }

    void startBenchmark(bool full, bool smoke) {
        cases_.clear();
        const std::array methods = {Method::FullRate, Method::Fixed2x2, Method::Fixed4x4, Method::Heuristic,
                                   Method::RadianceBudget, Method::FeatureBudget, Method::ParallelFeature, Method::DirectFullRate};
        const int blocks = smoke ? 1 : (full ? 3 : 2);
        const auto append = [&](const int w, const int h, const int lights, const int camera,
                                const float roughness, const float budget) {
            for (int block = 0; block < blocks; ++block) {
                const std::size_t start = (cases_.size()/methods.size() + block) % methods.size();
                for (std::size_t rank = 0; rank < methods.size(); ++rank) {
                    const std::size_t offset = block % 2 ? methods.size()-1-rank : rank;
                    cases_.push_back({methods[(start+offset)%methods.size()],w,h,lights,camera,roughness,budget,block});
                }
            }
        };
        if (smoke) {
            append(320,180,16,0,1.0f,0.025f);
            warmupFrames_ = 2; sampleFrames_ = 4;
        } else {
            for (const float budget : {0.010f,0.025f,0.050f}) {
                if (full) {
                    for (auto [w,h] : {std::pair{1280,720},std::pair{1920,1080}})
                        for (int lights : {16,64,256}) for (int camera=0; camera<3; ++camera)
                            append(w,h,lights,camera,1.0f,budget);
                    for (float roughness : {0.55f,1.6f}) append(1920,1080,256,2,roughness,budget);
                } else {
                    append(640,360,16,0,1.0f,budget);
                    append(640,360,64,1,1.0f,budget);
                    append(640,360,256,2,0.55f,budget);
                }
            }
            warmupFrames_ = full ? 12 : 5;
            sampleFrames_ = full ? 48 : 12;
        }

        const std::string profile = smoke ? "smoke" : (full ? "paper" : "quick");
        benchmarkOutput_ = options_.output;
        if (!runsBenchmarkFromCli()) {
            const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
            benchmarkOutput_ = std::filesystem::absolute(benchmarkOutput_ / (profile + "-" + std::to_string(stamp)));
            liveSettings_ = LiveSettings{camera_, method_, lightCount_, roughnessScale_, errorBudget_};
        }
        if (std::filesystem::exists(benchmarkOutput_) && !std::filesystem::is_empty(benchmarkOutput_))
            throw std::runtime_error("Choose an empty benchmark output directory");
        std::filesystem::create_directories(benchmarkOutput_);
        csv_.open(benchmarkOutput_ / "raw.csv");
        rawTiming_.open(benchmarkOutput_ / "timings.csv");
        manifest_.open(benchmarkOutput_ / "manifest.txt");
        if (!csv_ || !rawTiming_ || !manifest_) throw std::runtime_error("Could not open benchmark outputs");
        manifest_ << "profile=" << profile
                  << "\nplanned=" << cases_.size() << "\nwarmup=" << warmupFrames_ << "\nsamples=" << sampleFrames_
                  << "\nrenderer=" << reinterpret_cast<const char*>(glGetString(GL_RENDERER))
                  << "\nvendor=" << reinterpret_cast<const char*>(glGetString(GL_VENDOR))
                  << "\ndriver=" << reinterpret_cast<const char*>(glGetString(GL_VERSION)) << "\nseed=0\n";
        manifest_.flush();
        rawTiming_ << "case,block,method,sample,gpu_ns\n";
        csv_ << "method,width,height,lights,camera,roughness_scale,error_budget,gpu_median_ms,"
                "rmse,relative_rmse,gradient_rmse,max_abs,violation_rate,rate1,rate2,rate4,"
                "predictor_tile_rmse,false_safe_rate,max_underestimate,predictor_tiles,case,block,samples\n";
        benchmarkIndex_ = 0;
        applyCase(cases_.front());
        benchmarkRunning_ = true;
        benchmarkMessage_ = "Benchmark running.";
    }

    void finishBenchmark(bool completed) {
        benchmarkRunning_ = false;
        if (csv_.is_open()) csv_.close();
        if (rawTiming_.is_open()) rawTiming_.close();
        if (manifest_.is_open()) {
            manifest_ << "completed=" << (completed ? 1 : 0) << '\n';
            manifest_.close();
        }
        if (liveSettings_) {
            camera_ = liveSettings_->camera;
            method_ = liveSettings_->method;
            lightCount_ = liveSettings_->lights;
            roughnessScale_ = liveSettings_->roughness;
            errorBudget_ = liveSettings_->budget;
            liveSettings_.reset();
        }
        benchmarkMessage_ = completed ? "Benchmark complete." : "Benchmark stopped.";
        if (runsBenchmarkFromCli()) close();
    }

    void on_shutdown() override {
        if (benchmarkRunning_) finishBenchmark(false);
    }

    void benchmarkFrame() {
        const Case& c = cases_[benchmarkIndex_];
        // The G-buffer is frozen for the case. Query only the lighting dispatch and its barrier.
        GLuint query = 0;
        if (frameInCase_ >= warmupFrames_) glCreateQueries(GL_TIME_ELAPSED,1,&query);
        renderLighting(c.method, result_, c.width, c.height, false, query);
        if (query) {
            GLuint64 ns = 0;
            glGetQueryObjectui64v(query,GL_QUERY_RESULT,&ns);
            glDeleteQueries(1,&query);
            if (ns == 0) throw std::runtime_error("Invalid GPU elapsed result");
            rawTiming_ << benchmarkIndex_ << ',' << c.block << ',' << methodId(c.method) << ','
                       << timings_.size() << ',' << ns << '\n';
            timings_.push_back(static_cast<double>(ns)*1e-6);
        }

        if (!metricsCaptured_ && frameInCase_ == warmupFrames_) {
            const auto image = readRgba(result_.color(0), c.width, c.height);
            const auto rates = readR(result_.color(1), c.width, c.height);
            metrics_ = calculateMetrics(image, referenceImage_, rates, c.width, c.height, c.budget);
            metricsCaptured_ = true;
            if (c.method == Method::RadianceBudget && c.lights == 256 && c.cameraPreset == 2
                && c.block == 0 && c.roughnessScale == 1.0f && c.budget == 0.025f) {
                savePfm(benchmarkOutput_ / "captures" / "stress-method.pfm", result_.color(0), c.width, c.height);
                savePfm(benchmarkOutput_ / "captures" / "stress-reference.pfm", reference_.color(0), c.width, c.height);
            }
        }

        ++frameInCase_;
        if (frameInCase_ < warmupFrames_ + sampleFrames_) return;

        csv_ << methodId(c.method) << ',' << c.width << ',' << c.height << ',' << c.lights << ','
             << c.cameraPreset << ',' << c.roughnessScale << ',' << c.budget << ','
             << median(timings_) << ',' << metrics_.rmse << ',' << metrics_.relativeRmse << ','
             << metrics_.gradientRmse << ',' << metrics_.maxAbs << ',' << metrics_.violationRate << ','
             << metrics_.rate1 << ',' << metrics_.rate2 << ',' << metrics_.rate4 << ','
             << metrics_.predictorRmse << ',' << metrics_.falseSafeRate << ','
             << metrics_.maxUnderestimate << ',' << metrics_.predictorTiles << ',' << benchmarkIndex_ << ','
             << c.block << ',' << timings_.size() << '\n';
        csv_.flush();

        ++benchmarkIndex_;
        if (benchmarkIndex_ < cases_.size()) applyCase(cases_[benchmarkIndex_]);
        else finishBenchmark(true);
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

    struct LiveSettings {
        gfx::research::OrbitCamera camera;
        Method method;
        int lights;
        float roughness, budget;
    };
    std::optional<LiveSettings> liveSettings_;
    bool benchmarkRunning_ = false;
    std::filesystem::path benchmarkOutput_;
    std::string benchmarkMessage_;
    std::vector<Case> cases_;
    std::size_t benchmarkIndex_ = 0;
    int frameInCase_ = 0;
    int warmupFrames_ = 10;
    int sampleFrames_ = 30;
    std::vector<double> timings_;
    std::vector<float> referenceImage_;
    Metrics metrics_{};
    bool metricsCaptured_ = false;
    std::ofstream csv_, rawTiming_, manifest_;
};

LaunchOptions parseOptions(int argc, char** argv) {
    LaunchOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--publication-smoke") options.smoke = true;
        else if (arg == "--publication-quick") options.quick = true;
        else if (arg == "--publication-suite") options.full = true;
        else if (arg.starts_with("--output=")) options.output = std::string(arg.substr(9));
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    return RadianceRateResearch{parseOptions(argc, argv)}.run();
}
