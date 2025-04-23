import aiohttp
from datetime import datetime

ts = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def request_tracer():
    async def on_request_start(session, context, params):
        print(f'[{ts()}] on_request_start')

    async def on_connection_create_start(session, context, params):
        print(f'[{ts()}] on_request_start')

    async def on_request_redirect(session, context, params):
        print(f'[{ts()}] on_request_redirect')

    async def on_dns_resolvehost_start(session, context, params):
        print(f'[{ts()}] on_dns_resolvehost_start')

    async def on_dns_resolvehost_end(session, context, params):
        print(f'[{ts()}] on_dns_resolvehost_end')

    async def on_connection_create_end(session, context, params):
        print(f'[{ts()}] on_connection_create_end')

    async def on_request_chunk_sent(session, context, params):
        print(f'[{ts()}] on_request_chunk_sent')

    async def on_request_end(session, context, params):
        print(f'[{ts()}] on_request_end')

    async def on_response_chunk_received(session, context, params):
        print(f'[{ts()}] on_response_chunk_received')

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_redirect.append(on_request_redirect)
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
    trace_config.on_response_chunk_received.append(on_response_chunk_received)
    return trace_config
