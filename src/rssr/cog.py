"""Byte-budgeted HTTP range proxy for a single public Sentinel COG."""
import re,threading,time,urllib.request
from http.server import HTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse
from contextlib import contextmanager


@contextmanager
def bounded_cog(url,max_bytes=32_000_000):
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in ('sentinel-cogs.s3.us-west-2.amazonaws.com', 'e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com'):
        raise ValueError('Only the public Sentinel COG host is allowed')
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
        size=int(response.headers['Content-Length']); etag=response.headers.get('ETag')
    ledger={'bytes_received':0,'requests':0,'budget_bytes':max_bytes,'etag':etag,'source_size':size,'errors':[]}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_HEAD(self):
            self.send_response(200); self.send_header('Content-Length',str(size)); self.send_header('Accept-Ranges','bytes'); self.end_headers()
        def do_GET(self):
            try:
                match=re.fullmatch(r'bytes=(\d+)-(\d+)',self.headers.get('Range',''))
                if not match: raise ValueError('A bounded byte range is required')
                start,end=map(int,match.groups()); end=min(end,size-1); amount=end-start+1
                if start<0 or amount<1 or amount>8_000_000 or ledger['bytes_received']+amount>max_bytes:
                    raise ValueError('Range exceeds byte budget')
                headers={'Range':f'bytes={start}-{end}'}
                if etag: headers['If-Match']=etag
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as response:
                            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                                raise ValueError('Provider did not honor the exact range')
                            data=bytearray()
                            while len(data)<amount:
                                remaining=max_bytes-ledger['bytes_received']
                                if remaining<=0: raise ValueError('Transfer budget exhausted')
                                chunk=response.read(min(65536,amount-len(data),remaining))
                                if not chunk: raise OSError('Truncated range')
                                ledger['bytes_received']+=len(chunk)
                                if ledger['bytes_received']>max_bytes: raise ValueError('Transfer budget exhausted')
                                data.extend(chunk)
                            break
                    except OSError:
                        if attempt==2: raise
                        time.sleep(2**attempt)
                ledger['requests']+=1
                self.send_response(206); self.send_header('Content-Length',str(len(data))); self.send_header('Content-Range',f'bytes {start}-{end}/{size}'); self.end_headers(); self.wfile.write(data)
            except Exception as error:
                ledger['errors'].append(str(error)); self.send_error(502,str(error))
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}/asset.tif',ledger
    finally: server.shutdown(); server.server_close(); thread.join()
