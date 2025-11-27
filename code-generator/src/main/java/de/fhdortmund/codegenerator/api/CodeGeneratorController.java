package de.fhdortmund.codegenerator.api;

import de.fhdortmund.codegenerator.requests.Prompts;
import de.fhdortmund.codegenerator.service.InferenceService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.util.ArrayList;
import java.util.List;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/api")
public class CodeGeneratorController {

    private final InferenceService inferenceService;
    Logger logger = LogManager.getLogger(CodeGeneratorController.class);

    @Autowired
    public CodeGeneratorController(InferenceService inferenceService) {
        this.inferenceService = inferenceService;
    }

    @PostMapping("/process-request")
    public ResponseEntity<String> processReq(@Valid @NotNull @RequestBody Prompts prompt) {
        if (prompt.getPrompt() == null || prompt.getPrompt().isBlank()) {
            return ResponseEntity.ok("Invalid Request");
        }
        logger.info("Request received: {}", prompt.getPrompt());
        String msgId = inferenceService.addToQueue(prompt.getPrompt());
        if (!msgId.isBlank()) {
            return ResponseEntity.ok("Request: " + prompt.getPrompt() + " is added to the queue: " + msgId + " to process.");
        }
        return ResponseEntity.ok("Request is not processed: " + prompt.getPrompt());
    }

    @GetMapping("/results")
    public ResponseEntity<List<String>> getResults() {
        logger.info("Fetching all processed results for UI");
        List<String> results = inferenceService.fetchFromDB();
        if (results != null) {
            return ResponseEntity.ok(results);
        }
        return ResponseEntity.ok(new ArrayList<>());
    }
}
