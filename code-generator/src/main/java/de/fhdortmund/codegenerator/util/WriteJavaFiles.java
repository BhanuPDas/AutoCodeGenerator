package de.fhdortmund.codegenerator.util;

import com.google.googlejavaformat.java.FormatterException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Component;
import com.google.googlejavaformat.java.Formatter;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;
import java.nio.file.Paths;

@Component
public class WriteJavaFiles {

    private static final String PATH = "/root/java/generated-code";
    private final Path baseDir;
    Logger logger = LogManager.getLogger(WriteJavaFiles.class);

    public WriteJavaFiles() {
        this.baseDir = Paths.get(PATH);
    }

    public String saveFormattedJavaFile(String rawCode) throws IOException, FormatterException {
        logger.info("Checking if the directory exists: {}", PATH);
        ensureDirectoryExists(baseDir);
        Formatter formatter = new Formatter();
        logger.info("Formatting the file as Java code: {}", rawCode);
        String formattedCode = formatter.formatSource(rawCode);
        String fileName = generateUniqueFileName();
        Path filePath = baseDir.resolve(fileName);
        Files.writeString(filePath, formattedCode);
        logger.info("Java Code written to: {}", filePath.toAbsolutePath().toString());
        return formattedCode;
    }

    private void ensureDirectoryExists(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            Files.createDirectories(dir);
        }
    }

    private String generateUniqueFileName() {
        String timestamp = LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String random = UUID.randomUUID().toString().replace("-", "").substring(0, 8);
        return "Generated_" + timestamp + "_" + random + ".java";
    }
}
