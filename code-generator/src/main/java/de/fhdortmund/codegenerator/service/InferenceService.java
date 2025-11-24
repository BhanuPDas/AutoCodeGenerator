package de.fhdortmund.codegenerator.service;

import de.fhdortmund.codegenerator.requests.InferenceRequest;
import de.fhdortmund.codegenerator.response.InferenceResponse;
import de.fhdortmund.codegenerator.util.GenerateMetrics;
import de.fhdortmund.codegenerator.util.WriteJavaFiles;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;

import java.time.Duration;
import java.time.Instant;

@Service
public class InferenceService {

    private final RestTemplate rest;
    private final GenerateMetrics metrics;
    private final WriteJavaFiles jfiles;
    Logger logger = LogManager.getLogger(InferenceService.class);
    @Value("${inference.url}")
    private String inferenceUrl;

    @Autowired
    public InferenceService(RestTemplate rest, GenerateMetrics metrics, WriteJavaFiles jfiles) {
        this.rest = rest;
        this.metrics = metrics;
        this.jfiles = jfiles;
    }

    public String fetchInference(String prompt) {
        logger.info("Preparing request for inference for: {}", prompt);
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            InferenceRequest req = new InferenceRequest(prompt);
            HttpEntity<InferenceRequest> inferenceEntity = new HttpEntity<>(req, headers);
            Instant timeStart = Instant.now();
            ResponseEntity<InferenceResponse> response = rest.exchange(inferenceUrl, HttpMethod.POST, inferenceEntity,
                    InferenceResponse.class);
            Instant timeEnd = Instant.now();
            long latency = Duration.between(timeStart, timeEnd).toSeconds();
            logger.info("Latency of the request is: {} seconds", latency);
            if (response.getBody() != null) {
                String result = response.getBody().getResult();
                float inferenceTime = response.getBody().getInferenceTime();
                float tpms = response.getBody().getTpms();
                float nTokens = response.getBody().getNTokens();
                float gpuUtil = response.getBody().getGpuUtil();
                float memUtil = response.getBody().getMemUtil();
                float memUsed = response.getBody().getMemUsed();
                float totalMem = response.getBody().getTotalMem();
                logger.info("Inference response is: {} \n Inference time: {} \n Tokens per sec: {} \n No of Tokens generated: {} \n GPU Util%: {} \n Mem Engine Util%: {} \n Mem Used: {} \n Total Mem: {}",
                        result, inferenceTime, tpms, nTokens, gpuUtil, memUtil, memUsed, totalMem);
                metrics.updateMetrics(latency, inferenceTime, tpms, nTokens, gpuUtil, memUtil, memUsed, totalMem);
                return (jfiles.saveFormattedJavaFile(result));
            }
        } catch (Exception e) {
            logger.error("{} Error while fetching inference data for the prompt: {}", e.getMessage(), prompt);
            return "";
        }
        return "";
    }
}
