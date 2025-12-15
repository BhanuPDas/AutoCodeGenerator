package de.fhdortmund.codegenerator;

import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManager;
import org.apache.hc.core5.util.TimeValue;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Scope;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.client.RestTemplate;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import redis.clients.jedis.UnifiedJedis;

import java.util.concurrent.Executor;

@SpringBootApplication
public class CodeGeneratorApplication {

    @Value("${redis.url}")
    private String redisUrl;

    public static void main(String[] args) {
        SpringApplication.run(CodeGeneratorApplication.class, args);
    }

    @Bean
    @Scope(value = "singleton")
    public RestTemplate createRestTemplate(RestTemplateBuilder restBuilder) {
        PoolingHttpClientConnectionManager connectionManager = new PoolingHttpClientConnectionManager();
        connectionManager.setMaxTotal(5);
        connectionManager.setDefaultMaxPerRoute(5);
        CloseableHttpClient httpClient = HttpClients.custom()
                .setConnectionManager(connectionManager).evictExpiredConnections()
                .evictIdleConnections(TimeValue.ofSeconds(200))
                .build();
        HttpComponentsClientHttpRequestFactory requestFactory = new HttpComponentsClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(120000);
        requestFactory.setConnectTimeout(10000);
        requestFactory.setConnectionRequestTimeout(10000);
        return restBuilder.requestFactory(() -> requestFactory).build();
    }

    @Bean
    @Scope(value = "singleton")
    public UnifiedJedis stringRedisTemplate() {
        return new UnifiedJedis(redisUrl);
    }

    @Bean(name = "execThread")
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor exec = new ThreadPoolTaskExecutor();
        exec.setCorePoolSize(1);
        exec.setMaxPoolSize(1);
        exec.setQueueCapacity(100);
        exec.initialize();
        return exec;
    }


}
