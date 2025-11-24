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

    @PostMapping("/query")
    public ResponseEntity<String> query(@Valid @NotNull @RequestBody Prompts prompt) {
        if (prompt.getPrompt() == null || prompt.getPrompt().isBlank()) {
            return ResponseEntity.ok("Invalid Request");
        }
        logger.info("Request received: {}", prompt.getPrompt());
        String result = inferenceService.fetchInference(prompt.getPrompt());
        if (result.isBlank()) {
            return ResponseEntity.ok("No Result");
        }
        return ResponseEntity.ok(result);
    }
}
