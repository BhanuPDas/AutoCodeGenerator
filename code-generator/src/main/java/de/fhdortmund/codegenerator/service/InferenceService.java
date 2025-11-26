package de.fhdortmund.codegenerator.service;

import de.fhdortmund.codegenerator.entity.InferenceEntity;
import de.fhdortmund.codegenerator.repository.InferenceRepository;
import de.fhdortmund.codegenerator.util.GenerateMetrics;
import de.fhdortmund.codegenerator.util.WriteJavaFiles;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.web.client.RestTemplate;
import redis.clients.jedis.StreamEntryID;
import redis.clients.jedis.UnifiedJedis;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Service
public class InferenceService {

    private final RestTemplate rest;
    private final GenerateMetrics metrics;
    private final WriteJavaFiles jfiles;
    private final UnifiedJedis jedis;
    private final String streamKey = "PrompReq";
    private final String groupName = "llm";
    Logger logger = LogManager.getLogger(InferenceService.class);
    @Value("${inference.url}")
    private String inferenceUrl;
    private final InferenceRepository irepo;

    @Autowired
    public InferenceService(RestTemplate rest, GenerateMetrics metrics, WriteJavaFiles jfiles,
                            UnifiedJedis jedis, InferenceRepository irepo) {
        this.rest = rest;
        this.metrics = metrics;
        this.jfiles = jfiles;
        this.jedis = jedis;
        this.irepo = irepo;
    }

    public String addToQueue(String req) {
        Map<String, String> reqPrompt = new HashMap<>();
        reqPrompt.put("prompt", req);
        StreamEntryID id = jedis.xadd(streamKey, StreamEntryID.NEW_ENTRY, reqPrompt);
        if (id != null) {
            logger.info("Request has been added to the queue for further processing: {}", id.toString());
            return id.toString();
        }
        return "";
    }

    public List<String> fetchFromDB() {
        List<String> results = new ArrayList<>();
        try {
            List<InferenceEntity> resEntity = irepo.findBySendToUiEqualsIgnoreCase("no");
            for (InferenceEntity inferenceEntity : resEntity) {
                results.add(inferenceEntity.getResult());
                inferenceEntity.setSendToUi("yes");
            }
            irepo.saveAll(resEntity);

        } catch (Exception ex) {
            logger.error("Exception Occured while fetching results from DB: {}", ex.getMessage());
        }
        return results;
    }

    private void createGroup() {
        try {
            jedis.xgroupCreate(streamKey, groupName, StreamEntryID.XGROUP_LAST_ENTRY, true);
            logger.info("Redis group created successfully");
        } catch (Exception ignored) {
        }
    }

    @PostConstruct
    public void fetchPromptQueue() {
        String consumer = "c1";
        try {
            createGroup();
            ExecutorService service = Executors.newSingleThreadExecutor();
            service.submit(new ReadRedisQueue(jedis, groupName, consumer, streamKey, inferenceUrl, rest, metrics, jfiles, irepo));
        } catch (Exception ignored) {
        }
    }
}
