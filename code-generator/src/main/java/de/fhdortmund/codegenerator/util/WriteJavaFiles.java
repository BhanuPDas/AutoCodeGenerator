package de.fhdortmund.codegenerator.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Component;
import com.google.googlejavaformat.java.Formatter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.file.Paths;

@Component
public class WriteJavaFiles {

    private static final String PATH = "/root/java/generated-code";
    private final Path baseDir;
    Logger logger = LogManager.getLogger(WriteJavaFiles.class);

    public WriteJavaFiles() {
        this.baseDir = Paths.get(PATH);
    }

    public void saveFormattedJavaFile(int id, String rawCode) {
        try {
            logger.info("Checking if the directory exists: {}", PATH);
            ensureDirectoryExists(baseDir);
            Formatter formatter = new Formatter();
            logger.info("Formatting the file as Java code: {}", rawCode);
            StringBuilder genFileName = new StringBuilder("java-code-Q-");
            String extCode = extractCode(rawCode);
            if (!extCode.isBlank()) {
                boolean isValidClass = extCode.contains("class") || extCode.contains("interface") || extCode.contains("record") || extCode.contains("enum");
                if (!isValidClass) {
                    extCode = "public class GeneratedJavaCode {\n" + extCode + "\n}";
                }
                String formattedCode = formatter.formatSource(extCode);
                String fileName = genFileName.append(id).append(".java").toString();
                Path filePath = baseDir.resolve(fileName);
                Files.writeString(filePath, formattedCode);
                logger.info("Java Code written to: {}", filePath.toAbsolutePath().toString());
            } else {
                logger.error("No code found in response to write to a file");
            }
        } catch (Exception e) {
            logger.error("Error while writing Java code", e);
        }
    }

    private void ensureDirectoryExists(Path dir) {
        try {
            if (!Files.exists(dir)) {
                Files.createDirectories(dir);
            }
        } catch (Exception e) {
            logger.error("Unable to create directory: {}", e.getMessage());
        }
    }

    private String extractCode(String code) {
        try {
            Pattern pattern = Pattern.compile("```(.*?)```", Pattern.DOTALL);
            Matcher matcher = pattern.matcher(code);
            if (matcher.find()) {
                String extCode = matcher.group(1).trim();
                logger.info("Code is extracted from the text: {}", extCode);
                return extCode;
            } else {
                int startIdx = code.indexOf("package ");
                if (startIdx < 0) {
                    startIdx = code.indexOf("public class ");
                }
                if (startIdx >= 0) {
                    int endIdx = code.lastIndexOf("}");
                    if (endIdx >= 0 && endIdx >= startIdx) {
                        String extCode = code.substring(startIdx, endIdx + 1).trim();
                        logger.info("Code extracted from response: {}", extCode);
                        return extCode;
                    }
                }
            }
        } catch (Exception e) {
            logger.error("Error during extracting Code: {}", e.getMessage());
        }
        return "";
    }
}
