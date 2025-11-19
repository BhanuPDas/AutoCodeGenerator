package de.fhdortmund.codegenerator.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.fasterxml.jackson.annotation.JsonProperty;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class InferenceResponse {

    @JsonProperty("result")
    private String result;
    @JsonProperty("inference_time")
    private float inferenceTime;
    @JsonProperty("token_throughput")
    private float tpms;
    @JsonProperty("num_tokens")
    private float nTokens;
    @JsonProperty("gpu_util")
    private float gpuUtil;
    @JsonProperty("mem_util")
    private float memUtil;
    @JsonProperty("mem_used")
    private float memUsed;
    @JsonProperty("total_mem")
    private float totalMem;
}
