package de.fhdortmund.codegenerator.util;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.binder.MeterBinder;
import lombok.Getter;
import org.springframework.stereotype.Component;
import io.micrometer.common.lang.NonNull;

@Component
@Getter
public class GenerateMetrics implements MeterBinder {

    private volatile long latency;
    private volatile float inferenceTime;
    private volatile float tpms;
    private volatile float nTokens;
    private volatile float gpuUtil;
    private volatile float memUtil;
    private volatile float memUsed;
    private volatile float totalMem;

    public GenerateMetrics(MeterRegistry registry) {
        registry.gauge("latency", this, GenerateMetrics::getLatency);
        registry.gauge("inferenceTime", this, GenerateMetrics::getInferenceTime);
        registry.gauge("tokenPerSec", this, GenerateMetrics::getTpms);
        registry.gauge("NumTokens", this, GenerateMetrics::getNTokens);
        registry.gauge("gpuUtilization", this, GenerateMetrics::getGpuUtil);
        registry.gauge("memUtilization", this, GenerateMetrics::getMemUtil);
        registry.gauge("memUsed", this, GenerateMetrics::getMemUsed);
        registry.gauge("TotalMemory", this, GenerateMetrics::getTotalMem);

    }

    public void updateMetrics(long latency, float inferenceTime, float tpms, float nTokens,  float gpuUtil, float memUtil, float memUsed, float totalMem) {
        this.latency = latency;
        this.inferenceTime = inferenceTime;
        this.tpms = tpms;
        this.nTokens = nTokens;
        this.gpuUtil = gpuUtil;
        this.memUtil = memUtil;
        this.memUsed = memUsed;
        this.totalMem = totalMem;
    }

    @Override
    public void bindTo(@NonNull MeterRegistry meterRegistry) {
    }
}
