import SimpleHTTPServer
import CGIHTTPServer
import SocketServer

PORT = 8080

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

class ImageRequestHandler(CGIHTTPServer.CGIHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write("Hello World!")


    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write("Hello POST World!")

httpd = SocketServer.TCPServer(("", PORT), ImageRequestHandler)

print "serving at port", PORT
httpd.serve_forever()
