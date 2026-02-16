# ================ معماری توزیع شده برای میلیون‌ها کاربر ================

import asyncio
import aioredis
import aiohttp
from aiohttp import web
import uvloop
import sanic
from sanic import Sanic
from sanic.response import json
from sanic_cors import CORS
import uvicorn
import fastapi
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx
import grpc
from grpc import aio
import protobuf
import zeromq
import pyzmq
import nanomsg
import nng
import mqtt
from mqtt import MQTTClient
import paho.mqtt.client as mqtt
import kafka
from kafka import KafkaProducer, KafkaConsumer
import avro
from confluent_kafka import Producer, Consumer
import rabbitmq
from rabbitmq import RMQConnection
import pika
import nats
from nats.aio.client import Client as NATS
import stan
from stan.aio.client import Client as STAN
import pulsar
from pulsar import Client as PulsarClient
import activemq
from activemq import ActiveMQConnection
import artemis
from artemis import ArtemisClient
import qpid
from qpid import QpidClient
import hornetq
from hornetq import HornetQClient
import aws_sqs
from aws_sqs import SQSClient
import azure_servicebus
from azure.servicebus import ServiceBusClient
import gcp_pubsub
from google.cloud import pubsub_v1
import alibaba_rocketmq
from rocketmq.client import Producer, Consumer
import ibm_mq
from ibm_mq import MQClient
import solace
from solace import SolaceClient
import tibero
from tibero import TiberoClient
import altibase
from altibase import AltibaseClient
import cubrid
from cubrid import CUBRIDClient
import goldilocks
from goldilocks import GoldilocksClient
import machbase
from machbase import MachbaseClient

# ================ Load Balancer پیشرفته ================
import haproxy
from haproxy import HAProxyClient
import nginx
from nginx import NginxClient
import apache
from apache import ApacheClient
import traefik
from traefik import TraefikClient
import envoy
from envoy import EnvoyClient
import istio
from istio import IstioClient
import linkerd
from linkerd import LinkerdClient
import consul
from consul import ConsulClient
import etcd
from etcd import EtcdClient
import zookeeper
from kazoo.client import KazooClient
import eureka
from eureka import EurekaClient
import nacos
from nacos import NacosClient
import apollo
from apollo import ApolloClient

# ================ Redis Cluster برای کش میلیونی ================
import rediscluster
from rediscluster import RedisCluster
import redissharded
from redis_shard import RedisSharded
import redissentinel
from redis_sentinel import RedisSentinel
import redisbloom
from redisbloom import Client as BloomClient
import redistimeseries
from redistimeseries.client import Client as TimeSeriesClient
import redisgraph
from redisgraph import Graph
import redisearch
from redisearch import Client as SearchClient, IndexDefinition, TextField, NumericField
import redisjson
from redisjson import Client as JSONClient

# ================ Cassandra Cluster برای ذخیره‌سازی مقیاس‌پذیر ================
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
from cassandra.query import BatchStatement, PreparedStatement
from cassandra.concurrent import execute_concurrent

# ================ Elasticsearch Cluster برای جستجوی توزیع شده ================
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch.helpers import bulk, async_bulk, scan
from elasticsearch_dsl import Search, Q

# ================ ClickHouse برای آنالیز لحظه‌ای ================
from clickhouse_driver import Client as ClickHouseClient
from clickhouse_driver import connect as clickhouse_connect

# ================ TimescaleDB برای داده‌های زمانی ================
import psycopg2
from psycopg2.extras import execute_values
from timescaledb import TimescaleDB

# ================ InfluxDB برای متریک‌ها ================
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ================ Prometheus + Grafana برای مانیتورینگ ================
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
from prometheus_client.core import CollectorRegistry
import grafana_api
from grafana_api.grafana_face import GrafanaFace

# ================ Jaeger + Zipkin برای Tracing ================
from jaeger_client import Config
from jaeger_client.metrics.prometheus import PrometheusMetricsFactory
from opentracing import tracer
import zipkin
from py_zipkin import zipkin_span, zipkin_client_span

# ================ متغیرهای محیطی پیشرفته ================
import os
from dotenv import load_dotenv
load_dotenv()

# ================ تنظیمات کلاستر ================
class ClusterConfig:
    # Redis Cluster
    REDIS_NODES = [
        {'host': 'redis-1', 'port': 6379},
        {'host': 'redis-2', 'port': 6379},
        {'host': 'redis-3', 'port': 6379},
        {'host': 'redis-4', 'port': 6379},
        {'host': 'redis-5', 'port': 6379},
        {'host': 'redis-6', 'port': 6379},
    ]
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    
    # Cassandra Cluster
    CASSANDRA_HOSTS = ['cassandra-1', 'cassandra-2', 'cassandra-3']
    CASSANDRA_KEYSPACE = 'history_bot'
    
    # Elasticsearch Cluster
    ELASTICSEARCH_HOSTS = ['elastic-1:9200', 'elastic-2:9200', 'elastic-3:9200']
    
    # ClickHouse Cluster
    CLICKHOUSE_HOSTS = ['clickhouse-1', 'clickhouse-2', 'clickhouse-3']
    
    # Kafka Cluster
    KAFKA_BOOTSTRAP_SERVERS = ['kafka-1:9092', 'kafka-2:9092', 'kafka-3:9092']
    
    # RabbitMQ Cluster
    RABBITMQ_HOSTS = ['rabbit-1', 'rabbit-2', 'rabbit-3']
    
    # NATS Cluster
    NATS_SERVERS = ['nats://nats-1:4222', 'nats://nats-2:4222', 'nats://nats-3:4222']

# ================ سیستم کش توزیع شده ================
class DistributedCache:
    def __init__(self):
        self.redis_cluster = RedisCluster(
            startup_nodes=ClusterConfig.REDIS_NODES,
            password=ClusterConfig.REDIS_PASSWORD,
            decode_responses=True,
            max_connections_per_node=100,
            health_check_interval=30
        )
        
        # Redis Modules
        self.bloom = BloomClient(redis_cluster=self.redis_cluster)
        self.timeseries = TimeSeriesClient(redis_cluster=self.redis_cluster)
        self.graph = Graph('knowledge_graph', self.redis_cluster)
        self.search = SearchClient('idx_knowledge', conn=self.redis_cluster)
        self.json_client = JSONClient(redis_cluster=self.redis_cluster)
        
        # Local LRU Cache for hot data
        self.local_cache = {}
        self.local_cache_ttl = {}
        
    async def get(self, key, use_local=True):
        """دریافت از کش با fallback"""
        if use_local and key in self.local_cache:
            if self.local_cache_ttl[key] > time.time():
                return self.local_cache[key]
            else:
                del self.local_cache[key]
                del self.local_cache_ttl[key]
        
        # Try Redis Cluster
        try:
            value = await self.redis_cluster.get(key)
            if value and use_local:
                self.local_cache[key] = value
                self.local_cache_ttl[key] = time.time() + 60  # 60 seconds TTL
            return value
        except Exception as e:
            logger.error(f"Redis error: {e}")
            return None
    
    async def set(self, key, value, ttl=3600):
        """ذخیره در کش"""
        try:
            await self.redis_cluster.setex(key, ttl, value)
            self.local_cache[key] = value
            self.local_cache_ttl[key] = time.time() + min(ttl, 60)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def mget(self, keys):
        """دریافت چندتایی"""
        try:
            return await self.redis_cluster.mget(keys)
        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            return [None] * len(keys)
    
    async def incr(self, key, amount=1):
        """افزایش مقدار"""
        try:
            return await self.redis_cluster.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis incr error: {e}")
            return 0

# ================ Message Queue برای پردازش غیرهمزمان ================
class MessageQueue:
    def __init__(self):
        # Kafka for high throughput
        self.kafka_producer = Producer({
            'bootstrap.servers': ','.join(ClusterConfig.KAFKA_BOOTSTRAP_SERVERS),
            'client.id': 'history-bot-producer',
            'acks': 'all',
            'retries': 10,
            'compression.type': 'snappy',
            'batch.size': 16384,
            'linger.ms': 5
        })
        
        # RabbitMQ for RPC
        self.rabbit_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=ClusterConfig.RABBITMQ_HOSTS[0],
                credentials=pika.PlainCredentials('guest', 'guest')
            )
        )
        self.rabbit_channel = self.rabbit_connection.channel()
        
        # NATS for real-time
        self.nats_client = NATS()
        
        # ZeroMQ for low-latency
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        
    async def publish_chat_message(self, message):
        """انتشار پیام چت به صف"""
        # Kafka for persistence
        self.kafka_producer.produce(
            'chat-messages',
            key=str(message['user_id']).encode(),
            value=json.dumps(message).encode(),
            callback=self.delivery_report
        )
        
        # NATS for real-time
        await self.nats_client.publish(
            'chat.messages',
            json.dumps(message).encode()
        )
        
    async def publish_analytics(self, data):
        """انتشار داده‌های تحلیلی"""
        self.kafka_producer.produce(
            'analytics',
            value=json.dumps(data).encode(),
            callback=self.delivery_report
        )
    
    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()}")

# ================ Connection Pool برای دیتابیس ================
class DatabasePool:
    def __init__(self):
        # Cassandra Session Pool
        self.cassandra_cluster = Cluster(
            ClusterConfig.CASSANDRA_HOSTS,
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            protocol_version=4,
            connection_class='asyncio'
        )
        self.cassandra_session = self.cassandra_cluster.connect()
        
        # Elasticsearch Async Client
        self.elasticsearch = AsyncElasticsearch(ClusterConfig.ELASTICSEARCH_HOSTS)
        
        # ClickHouse Pool
        self.clickhouse_clients = [
            ClickHouseClient(host) for host in ClusterConfig.CLICKHOUSE_HOSTS
        ]
        
        # PostgreSQL/TimescaleDB Pool
        self.postgres_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,
            host='timescaledb',
            database='history_bot',
            user='postgres',
            password='postgres'
        )
        
        # InfluxDB Client
        self.influxdb = InfluxDBClient(
            url='http://influxdb:8086',
            token='my-token',
            org='history-bot'
        )
        
    async def execute_cassandra(self, query, params=None):
        """اجرای کوئری Cassandra"""
        try:
            prepared = self.cassandra_session.prepare(query)
            result = await self.cassandra_session.execute_async(prepared, params)
            return result
        except Exception as e:
            logger.error(f"Cassandra error: {e}")
            return None
    
    async def search_elasticsearch(self, index, query):
        """جستجو در Elasticsearch"""
        try:
            result = await self.elasticsearch.search(
                index=index,
                body=query,
                size=100
            )
            return result['hits']['hits']
        except Exception as e:
            logger.error(f"Elasticsearch error: {e}")
            return []

# ================ هسته اصلی هوش مصنوعی با قابلیت توزیع ================
class DistributedAI:
    def __init__(self):
        self.cache = DistributedCache()
        self.mq = MessageQueue()
        self.db = DatabasePool()
        self.models = {}
        self.load_models()
        
        # آمار لحظه‌ای
        self.stats = {
            'active_users': Gauge('active_users', 'Number of active users'),
            'requests_per_second': Counter('requests_total', 'Total requests'),
            'response_time': Histogram('response_time_seconds', 'Response time'),
            'knowledge_size': Gauge('knowledge_size', 'Total knowledge entries')
        }
        
    def load_models(self):
        """بارگذاری مدل‌های ML در حافظه"""
        # TensorFlow Serving or PyTorch Serve
        self.models['qa'] = self.load_qa_model()
        self.models['embedding'] = self.load_embedding_model()
        self.models['classification'] = self.load_classification_model()
        
    async def get_answer_distributed(self, question, user_id=None):
        """پاسخگویی توزیع شده با قابلیت مقیاس‌پذیری"""
        start_time = time.time()
        
        # بررسی کش
        cache_key = f"answer:{hashlib.md5(question.encode()).hexdigest()}"
        cached_answer = await self.cache.get(cache_key)
        if cached_answer:
            self.stats['response_time'].observe(time.time() - start_time)
            return json.loads(cached_answer)
        
        # جستجو در Elasticsearch
        search_results = await self.search_knowledge(question)
        
        if search_results:
            # استفاده از مدل QA
            answer = await self.generate_answer(question, search_results)
            
            # ذخیره در کش
            await self.cache.set(cache_key, json.dumps(answer), ttl=3600)
            
            # انتشار به Kafka برای تحلیل
            await self.mq.publish_analytics({
                'type': 'answer',
                'question': question,
                'answer': answer,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            })
            
            self.stats['response_time'].observe(time.time() - start_time)
            return answer
        else:
            # ثبت سوال بی‌پاسخ
            await self.record_unanswered(question, user_id)
            return None
    
    async def search_knowledge(self, question):
        """جستجوی توزیع شده در دانش"""
        # Elasticsearch query
        es_query = {
            "query": {
                "multi_match": {
                    "query": question,
                    "fields": ["question^3", "answer^2", "tags^1.5"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": 10,
            "sort": ["_score"]
        }
        
        results = await self.db.search_elasticsearch('knowledge', es_query)
        return results
    
    async def generate_answer(self, question, contexts):
        """تولید پاسخ با استفاده از مدل"""
        # اینجا می‌تونی از مدل‌های Transformer استفاده کنی
        # مثلاً BERT, GPT, T5 و ...
        
        best_context = contexts[0]['_source']['answer']
        
        # QA Pipeline
        if self.models.get('qa'):
            answer = self.models['qa']({
                'question': question,
                'context': best_context
            })
            return answer['answer']
        
        return best_context
    
    async def record_unanswered(self, question, user_id):
        """ثبت سوال بی‌پاسخ در Cassandra"""
        query = """
        INSERT INTO unanswered_questions (id, question, user_id, timestamp, status)
        VALUES (uuid(), ?, ?, ?, 'pending')
        """
        params = [question, user_id, datetime.now()]
        await self.db.execute_cassandra(query, params)
        
        # انتشار به Kafka
        await self.mq.publish_chat_message({
            'type': 'unanswered',
            'question': question,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })

# ================ WebSocket Manager برای میلیون‌ها اتصال همزمان ================
class WebSocketManager:
    def __init__(self):
        self.active_connections = {}
        self.user_sessions = {}
        self.room_sessions = defaultdict(set)
        self.stats = Gauge('websocket_connections', 'Number of WebSocket connections')
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = {
            'websocket': websocket,
            'connected_at': datetime.now(),
            'last_activity': datetime.now()
        }
        self.stats.inc()
        
    async def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            self.stats.dec()
            
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id]['websocket'].send_text(message)
            
    async def broadcast(self, message: str):
        for user_id in self.active_connections:
            try:
                await self.active_connections[user_id]['websocket'].send_text(message)
            except:
                await self.disconnect(user_id)
                
    def get_stats(self):
        return {
            'total_connections': len(self.active_connections),
            'connections_by_minute': self.get_connections_by_minute()
        }

# ================ FastAPI Application ================
app = FastAPI(title="History Bot Ultra", version="6.9.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
ai = DistributedAI()
ws_manager = WebSocketManager()

# ================ WebSocket Endpoint برای چت زنده ================
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await ws_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # پردازش پیام
            answer = await ai.get_answer_distributed(
                question=message['question'],
                user_id=user_id
            )
            
            # ارسال پاسخ
            await websocket.send_json({
                'type': 'answer',
                'data': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            # به‌روزرسانی آخرین فعالیت
            ws_manager.active_connections[user_id]['last_activity'] = datetime.now()
            
    except WebSocketDisconnect:
        await ws_manager.disconnect(user_id)

# ================ API Endpoints ================
@app.get("/api/stats")
async def get_stats():
    """دریافت آمار لحظه‌ای"""
    return {
        'websocket_connections': ws_manager.get_stats(),
        'cache_stats': {
            'size': len(ai.cache.local_cache),
            'hits': 0,  # باید از Redis آمار بگیری
            'misses': 0
        },
        'knowledge_stats': {
            'total': await get_knowledge_count(),
            'categories': await get_category_stats()
        }
    }

@app.get("/api/conversations/stream")
async def stream_conversations(request: Request):
    """Streaming endpoint برای نمایش قطاری مکالمات"""
    async def event_generator():
        client = ai.mq.kafka_consumer
        client.subscribe(['chat-messages'])
        
        while True:
            msg = client.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
                
            yield f"data: {msg.value().decode()}\n\n"
            
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

# ================ پنل مدیریت قطاری ================
@app.get("/admin/dashboard")
async def admin_dashboard(request: Request):
    """پنل مدیریت با نمایش قطاری مکالمات"""
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "stats": await get_stats(),
        "recent_conversations": await get_recent_conversations(limit=100)
    })

@app.get("/admin/api/conversations/realtime")
async def admin_conversations_realtime():
    """API برای نمایش قطاری مکالمات با Server-Sent Events"""
    async def event_generator():
        # اتصال به Kafka برای دریافت لحظه‌ای مکالمات
        consumer = KafkaConsumer(
            'chat-messages',
            'answers',
            'unanswered',
            bootstrap_servers=ClusterConfig.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        
        for message in consumer:
            yield {
                'event': message.topic,
                'data': message.value,
                'timestamp': datetime.now().isoformat()
            }
    
    return EventSourceResponse(event_generator())

# ================ HTML Template برای پنل مدیریت قطاری ================
ADMIN_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>پنل مدیریت پیشرفته - تاریخ‌دان هوشمند</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@100..900&display=swap');
        body { font-family: 'Vazirmatn', sans-serif; }
        .conversation-stream {
            height: 600px;
            overflow-y: auto;
            scroll-behavior: smooth;
        }
        .conversation-item {
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .stat-card {
            transition: all 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-6 mb-8 text-white">
            <div class="flex justify-between items-center">
                <h1 class="text-3xl font-bold">🤖 پنل مدیریت تاریخ‌دان هوشمند</h1>
                <div class="flex gap-4">
                    <div class="bg-white/20 rounded-lg px-4 py-2">
                        <span class="text-sm">اتصال‌های فعال</span>
                        <span class="text-2xl font-bold mr-2" id="active-connections">0</span>
                    </div>
                    <div class="bg-white/20 rounded-lg px-4 py-2">
                        <span class="text-sm">درخواست/ثانیه</span>
                        <span class="text-2xl font-bold mr-2" id="requests-per-second">0</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Stats Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="stat-card bg-white rounded-xl p-6 shadow-lg">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-500 text-sm">کل دانش</p>
                        <p class="text-3xl font-bold" id="total-knowledge">0</p>
                    </div>
                    <div class="bg-blue-100 p-3 rounded-full">
                        <svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
                        </svg>
                    </div>
                </div>
            </div>
            
            <div class="stat-card bg-white rounded-xl p-6 shadow-lg">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-500 text-sm">مکالمات امروز</p>
                        <p class="text-3xl font-bold" id="today-conversations">0</p>
                    </div>
                    <div class="bg-green-100 p-3 rounded-full">
                        <svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                        </svg>
                    </div>
                </div>
            </div>
            
            <div class="stat-card bg-white rounded-xl p-6 shadow-lg">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-500 text-sm">سوالات بی‌پاسخ</p>
                        <p class="text-3xl font-bold" id="unanswered-count">0</p>
                    </div>
                    <div class="bg-red-100 p-3 rounded-full">
                        <svg class="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                </div>
            </div>
            
            <div class="stat-card bg-white rounded-xl p-6 shadow-lg">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-500 text-sm">میانگین زمان پاسخ</p>
                        <p class="text-3xl font-bold" id="avg-response-time">0</p>
                    </div>
                    <div class="bg-purple-100 p-3 rounded-full">
                        <svg class="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-xl p-6 shadow-lg">
                <h3 class="text-lg font-semibold mb-4">📊 فعالیت در ۲۴ ساعت گذشته</h3>
                <canvas id="activity-chart"></canvas>
            </div>
            <div class="bg-white rounded-xl p-6 shadow-lg">
                <h3 class="text-lg font-semibold mb-4">🎯 دسته‌بندی سوالات</h3>
                <canvas id="categories-chart"></canvas>
            </div>
        </div>
        
        <!-- Real-time Conversation Stream -->
        <div class="bg-white rounded-xl shadow-lg overflow-hidden">
            <div class="bg-gray-50 px-6 py-4 border-b">
                <div class="flex justify-between items-center">
                    <h3 class="text-lg font-semibold">💬 جریان زنده مکالمات</h3>
                    <div class="flex gap-2">
                        <span class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm flex items-center">
                            <span class="w-2 h-2 bg-green-500 rounded-full ml-2"></span>
                            live
                        </span>
                        <span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm" id="messages-count">0 پیام</span>
                    </div>
                </div>
            </div>
            <div class="conversation-stream p-4" id="conversation-stream">
                <!-- Conversations will appear here -->
            </div>
        </div>
    </div>
    
    <script>
        // اتصال به WebSocket برای دریافت لحظه‌ای
        const socket = io('ws://localhost:5000/admin/ws');
        const conversationStream = document.getElementById('conversation-stream');
        let messageCount = 0;
        
        socket.on('connect', function() {
            console.log('✅ Connected to admin panel');
        });
        
        socket.on('new_conversation', function(data) {
            addConversationToStream(data);
            updateStats(data);
        });
        
        socket.on('new_answer', function(data) {
            addAnswerToStream(data);
        });
        
        socket.on('unanswered_question', function(data) {
            addUnansweredToStream(data);
        });
        
        function addConversationToStream(data) {
            messageCount++;
            document.getElementById('messages-count').innerText = messageCount + ' پیام';
            
            const item = document.createElement('div');
            item.className = 'conversation-item bg-gray-50 rounded-lg p-4 mb-3 border-r-4 border-blue-500';
            item.innerHTML = `
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <div class="flex items-center gap-2 mb-2">
                            <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">سوال</span>
                            <span class="text-sm text-gray-500">${data.user_id || 'ناشناس'}</span>
                            <span class="text-xs text-gray-400">${new Date(data.timestamp).toLocaleTimeString('fa-IR')}</span>
                        </div>
                        <p class="text-gray-800">${data.question}</p>
                    </div>
                </div>
            `;
            
            conversationStream.insertBefore(item, conversationStream.firstChild);
            
            // محدودیت تعداد پیام‌ها
            if (conversationStream.children.length > 100) {
                conversationStream.removeChild(conversationStream.lastChild);
            }
        }
        
        function addAnswerToStream(data) {
            const item = document.createElement('div');
            item.className = 'conversation-item bg-gray-50 rounded-lg p-4 mb-3 border-r-4 border-green-500';
            item.innerHTML = `
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <div class="flex items-center gap-2 mb-2">
                            <span class="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">پاسخ</span>
                            <span class="text-sm text-gray-500">بات</span>
                            <span class="text-xs text-gray-400">${new Date(data.timestamp).toLocaleTimeString('fa-IR')}</span>
                            <span class="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">${Math.round(data.confidence * 100)}% اطمینان</span>
                        </div>
                        <p class="text-gray-800">${data.answer}</p>
                        <p class="text-xs text-gray-500 mt-1">به سوال: ${data.question}</p>
                    </div>
                </div>
            `;
            
            conversationStream.insertBefore(item, conversationStream.firstChild);
        }
        
        function addUnansweredToStream(data) {
            const item = document.createElement('div');
            item.className = 'conversation-item bg-gray-50 rounded-lg p-4 mb-3 border-r-4 border-red-500';
            item.innerHTML = `
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <div class="flex items-center gap-2 mb-2">
                            <span class="px-2 py-1 bg-red-100 text-red-800 rounded text-xs">بی‌پاسخ</span>
                            <span class="text-sm text-gray-500">${data.user_id || 'ناشناس'}</span>
                            <span class="text-xs text-gray-400">${new Date(data.timestamp).toLocaleTimeString('fa-IR')}</span>
                        </div>
                        <p class="text-gray-800">${data.question}</p>
                        <button onclick="answerQuestion('${data.id}')" class="mt-2 px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600">
                            ✍️ پاسخ به این سوال
                        </button>
                    </div>
                </div>
            `;
            
            conversationStream.insertBefore(item, conversationStream.firstChild);
        }
        
        // نمودارها
        const ctx1 = document.getElementById('activity-chart').getContext('2d');
        new Chart(ctx1, {
            type: 'line',
            data: {
                labels: Array.from({length: 24}, (_, i) => i + ':00'),
                datasets: [{
                    label: 'تعداد سوالات',
                    data: Array.from({length: 24}, () => Math.floor(Math.random() * 100)),
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        const ctx2 = document.getElementById('categories-chart').getContext('2d');
        new Chart(ctx2, {
            type: 'doughnut',
            data: {
                labels: ['ایران باستان', 'اسلامی', 'معاصر', 'جهان', 'علمی'],
                datasets: [{
                    data: [45, 25, 15, 10, 5],
                    backgroundColor: [
                        'rgb(59, 130, 246)',
                        'rgb(16, 185, 129)',
                        'rgb(245, 158, 11)',
                        'rgb(239, 68, 68)',
                        'rgb(139, 92, 246)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        
        // به‌روزرسانی آمار
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('active-connections').innerText = stats.websocket_connections.total_connections;
                document.getElementById('total-knowledge').innerText = stats.knowledge_stats.total;
                document.getElementById('avg-response-time').innerText = (stats.response_time?.avg || 0).toFixed(2) + 's';
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        // به‌روزرسانی هر ۵ ثانیه
        setInterval(updateStats, 5000);
        
        // Streaming با SSE
        const eventSource = new EventSource('/admin/api/conversations/realtime');
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.event === 'chat-messages') {
                addConversationToStream(data.data);
            }
        };
    </script>
</body>
</html>
"""

# ================ Kubernetes Deployment برای مقیاس‌پذیری ================
K8S_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: history-bot
spec:
  replicas: 100
  selector:
    matchLabels:
      app: history-bot
  template:
    metadata:
      labels:
        app: history-bot
    spec:
      containers:
      - name: history-bot
        image: history-bot:latest
        ports:
        - containerPort: 5000
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: history-bot-service
spec:
  selector:
    app: history-bot
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: history-bot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: history-bot
  minReplicas: 10
  maxReplicas: 1000
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 1000
"""

# ================ اجرای برنامه ================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        workers=8,  # تعداد workerها بر اساس CPU
        loop="uvloop",
        http="httptools",
        limit_concurrency=10000,
        backlog=2048,
        timeout_keep_alive=5
          )
