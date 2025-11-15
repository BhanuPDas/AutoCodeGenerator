package de.fhdortmund.codegenerator.api;

import de.fhdortmund.codegenerator.requests.Prompts;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@CrossOrigin(origins = "http://localhost:5173/")
@RequestMapping("/api")
public class CodeGeneratorController {
    private int rNo = 0;

    @PostMapping("/query")
    public ResponseEntity<String> query(@RequestBody Prompts prompt) {
        System.out.println("Request received: " + prompt.getPrompt());
        return ResponseEntity.ok("Hello" + (++rNo));
    }
}
