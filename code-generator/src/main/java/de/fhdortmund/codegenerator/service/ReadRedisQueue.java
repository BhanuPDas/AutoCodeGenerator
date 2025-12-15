package de.fhdortmund.codegenerator.service;

import com.google.gson.Gson;
import de.fhdortmund.codegenerator.entity.InferenceEntity;
import de.fhdortmund.codegenerator.repository.InferenceRepository;
import de.fhdortmund.codegenerator.requests.InferenceRequest;
import de.fhdortmund.codegenerator.requests.Prompts;
import de.fhdortmund.codegenerator.response.InferenceResponse;
import de.fhdortmund.codegenerator.util.GenerateMetrics;
import de.fhdortmund.codegenerator.util.WriteJavaFiles;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;
import redis.clients.jedis.StreamEntryID;
import redis.clients.jedis.UnifiedJedis;
import redis.clients.jedis.params.XReadGroupParams;
import redis.clients.jedis.resps.StreamEntry;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ReadRedisQueue implements Runnable {

    private final UnifiedJedis jedis;
    private final String groupName;
    private final String consumer;
    private final String streamKey;
    private final String inferenceUrl;
    private final RestTemplate rest;
    private final GenerateMetrics metrics;
    private final WriteJavaFiles jfiles;
    private final InferenceRepository irepo;
    Logger logger = LogManager.getLogger(ReadRedisQueue.class);

    public ReadRedisQueue(UnifiedJedis jedis, String groupName, String consumer,
                          String streamKey, String inferenceUrl, RestTemplate rest,
                          GenerateMetrics metrics, WriteJavaFiles jfiles, InferenceRepository irepo) {
        this.jedis = jedis;
        this.groupName = groupName;
        this.consumer = consumer;
        this.streamKey = streamKey;
        this.inferenceUrl = inferenceUrl;
        this.rest = rest;
        this.metrics = metrics;
        this.jfiles = jfiles;
        this.irepo = irepo;
    }

    @Override
    public void run() {
        logger.info("Attempting to fetch queued prompts from Redis..");
        try {
            XReadGroupParams params = new XReadGroupParams().count(1).block(0);
            Map<String, StreamEntryID> bID = new HashMap<>();
            bID.put(streamKey, new StreamEntryID("0-0"));
            List<Map.Entry<String, List<StreamEntry>>> recordsPen = jedis.xreadGroup(groupName, consumer, params, bID);
            if (recordsPen!= null && !recordsPen.isEmpty()) {
                processStreamMessage(recordsPen);
            }
            while(!Thread.currentThread().isInterrupted()) {
                pollToFetchPrompt();
            }
        } catch (Exception e) {
            logger.error("Error while trying to read pending records from redis: {}", e.getMessage());
        }
    }

    private void saveResultInDB(List<InferenceEntity> iEntityList) {
        try {
            if (iEntityList != null && !iEntityList.isEmpty()) {
                logger.info("Saving results into DB.");
                irepo.saveAll(iEntityList);
            }
        } catch (Exception e) {
            logger.error("Error while saving results to DB: {}", e.getMessage());
        }
    }

    private void processStreamMessage(List<Map.Entry<String, List<StreamEntry>>> records) {
        List<InferenceEntity> iEntityList = new ArrayList<>();
        if (records != null) {
            for (Map.Entry<String, List<StreamEntry>> streamData : records) {
                String stream = streamData.getKey();
                logger.info("Fetching {} record from redis...", stream);
                List<StreamEntry> entries = streamData.getValue();
                for (StreamEntry entry : entries) {
                    StreamEntryID messageId = entry.getID();
                    Map<String, String> fields = entry.getFields();
                    String prompt = fields.get("prompt");
                    try {
                        logger.info("Received message ID = {} & Prompt: {}", messageId.toString(), prompt);
                        Gson gson = new Gson();
                        Prompts promptReq = gson.fromJson(prompt, Prompts.class);

                        HttpHeaders headers = new HttpHeaders();
                        headers.setContentType(MediaType.APPLICATION_JSON);
                        InferenceRequest req = new InferenceRequest(promptReq.getPrompt());
                        HttpEntity<InferenceRequest> inferenceEntity = new HttpEntity<>(req, headers);
                        Instant timeStart = Instant.now();
                        ResponseEntity<InferenceResponse> response = rest.exchange(inferenceUrl, HttpMethod.POST, inferenceEntity,
                                InferenceResponse.class);
                        Instant timeEnd = Instant.now();
                        double latency = (Duration.between(timeStart, timeEnd).toMillis()) / 1000.0;
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
                            logger.info("Inference response is: {} \n Latency: {} \n Inference time: {} \n Tokens per sec: {} \n No of Tokens generated: {} \n GPU Util%: {} \n Mem Engine Util%: {} \n Mem Used: {} \n Total Mem: {}",
                                    result, latency, inferenceTime, tpms, nTokens, gpuUtil, memUtil, memUsed, totalMem);
                            metrics.updateMetrics(latency, inferenceTime, tpms, nTokens, gpuUtil, memUtil, memUsed, totalMem);
                            jfiles.saveFormattedJavaFile(promptReq.getId(), result);
                            InferenceEntity iEntity = new InferenceEntity();
                            iEntity.setMsgId(messageId.toString());
                            iEntity.setPrompt(promptReq.getPrompt());
                            iEntity.setResult(result);
                            iEntity.setSentToUi("No");
                            iEntityList.add(iEntity);
                        }

                        jedis.xack(streamKey, groupName, messageId);
                    } catch (Exception e) {
                        logger.error("Error while processing ID: {} prompt: {} from redis: {}", messageId.toString(), prompt, e.getMessage());
                    }
                }
            }
        }
        saveResultInDB(iEntityList);
    }

    private void pollToFetchPrompt() {
        XReadGroupParams params = new XReadGroupParams().count(1).block(2);
        Map<String, StreamEntryID> mID = new HashMap<>();
        mID.put(streamKey, StreamEntryID.XREADGROUP_UNDELIVERED_ENTRY);
        try {
            List<Map.Entry<String, List<StreamEntry>>> records = jedis.xreadGroup(groupName, consumer, params, mID);
            if (records != null && !records.isEmpty()) {
                processStreamMessage(records);
            }
        } catch (Exception e) {
            logger.error("Error while fetching the new prompt from redis: {}", e.getMessage());
        }
    }
}
