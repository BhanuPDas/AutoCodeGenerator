package de.fhdortmund.codegenerator;

import org.apache.hc.client5.http.ConnectionKeepAliveStrategy;
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
import org.springframework.web.client.RestTemplate;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import redis.clients.jedis.UnifiedJedis;

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
        PoolingHttpClientConnectionManager cm = new PoolingHttpClientConnectionManager();
        cm.setMaxTotal(5);
        cm.setDefaultMaxPerRoute(5);
        ConnectionKeepAliveStrategy keepAliveStrategy = (response, context) ->
                TimeValue.ofSeconds(60);
        CloseableHttpClient httpClient = HttpClients.custom()
                .setConnectionManager(cm)
                .setKeepAliveStrategy(keepAliveStrategy)
                .setConnectionReuseStrategy((req, resp, ctx) -> true)
                .evictExpiredConnections()
                .evictIdleConnections(TimeValue.ofSeconds(5))
                .build();

        HttpComponentsClientHttpRequestFactory factory = new HttpComponentsClientHttpRequestFactory(httpClient);
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(300000);
        factory.setConnectionRequestTimeout(5000);
        return restBuilder.requestFactory(() -> factory).build();
    }

    @Bean
    @Scope(value = "singleton")
    public UnifiedJedis stringRedisTemplate() {
        return new UnifiedJedis(redisUrl);
    }

}
