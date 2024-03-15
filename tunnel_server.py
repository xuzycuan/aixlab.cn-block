# -*- coding: UTF-8 -*-
import socket
import struct
import select
import traceback
import time

from contextlib import closing


class TunnelServer():
    def __init__(self):
        self.listen_fd = 0
        self.inputs = [ ]
        self.outputs = [ ]
        self.client_buffer = { }
        self.all_clients = { }
        self.named_clients = { }
        self.event_handler = None

    def run(self, host, port, event_handler=None):
        self.event_handler = event_handler

        s = socket.socket()

        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(5)

        self.listen_fd = s

        self.inputs.append(s)

        while True:
            try:
                list_r, list_w, list_e = select.select(self.inputs, self.outputs, self.inputs)

                if self.listen_fd in list_r:
                    sock, _ = self.listen_fd.accept()

                    self.client_buffer[sock] = bytearray()
                    self.all_clients[sock] = ''

                    self.inputs.append(sock)

                    print('==> New client:', sock)

            except:
                traceback.print_exc()

            self._process_read(list_r)
            self._process_write(list_w)
            self._process_except(list_e)

            time.sleep(0.02)

        print('==> Tunnel quit.')

    def stop(self):
        for sock in self.inputs:
            try:
                sock.close()

            except:
                traceback.print_exc()

        for sock, buffer in self.client_buffer.items():
            buffer.clear()

        self.listen_fd = 0
        self.inputs = [ ]
        self.outputs = [ ]
        self.client_buffer = { }
        self.all_clients = { }
        self.named_clients = { }
        self.event_handler = None

    def send(self, username, action, data):
        _action = bytes(action, "utf-8")
        _data = bytes(data, "utf-8")
        _alen = len(_action)
        _dlen = len(_data)

        if username in self.named_clients:
            sock = self.named_clients[username]
            head = struct.pack('>2I', 0x722231, struct.calcsize('>I') * 3 + _alen + _dlen)
            format = '>2I%isI%is' %(_alen, _dlen)
            packet = struct.pack(format, 2, _alen, _action, _dlen, _data)

            try:
                sock.sendall(head + packet)

            except:
                traceback.print_exc()

    def send_file(self, username, action, data, filename):
        _action = bytes(action, "utf-8")
        _data = bytes(data, "utf-8")
        _filedata = self.read_file(filename)
        _alen = len(_action)
        _dlen = len(_data)
        _flen = len(_filedata)

        if username in self.named_clients:
            sock = self.named_clients[username]
            head = struct.pack('>2I', 0x722231, struct.calcsize('>I') * 4 + _alen + _dlen + _flen)
            format = '>2I%isI%isI%is' %(_alen, _dlen, _flen)
            packet = struct.pack(format, 3, _alen, _action, _dlen, _data, _flen, _filedata)

            try:
                sock.sendall(head + packet)

            except:
                traceback.print_exc()

    def read_file(self, filename):
        try:
            with closing(open(filename, 'rb')) as f:
                return f.read()

        except:
            traceback.print_exc()

        return b''

    def write_file(self, filename, data):
        try:
            with closing(open(filename, 'wb')) as f:
                return f.write(data)

        except:
            traceback.print_exc()

        return 0

    def _packet_available(self, buffer):
        datas = bytes(buffer)
        datalen = len(datas)
        headlen = struct.calcsize('>2I')

        if datalen > headlen:
            remain = datalen - headlen
            sign, pklen = struct.unpack('>2I', datas[0:headlen])

            return sign == 0x722231 and pklen <= remain

        return False

    def _packet_parse(self, buffer):
        datas = bytes(buffer)
        headlen = struct.calcsize('>2I')
        ilen = struct.calcsize('>I')
        offset = 0
        offlen = headlen + ilen

        sign, pklen, pcount = struct.unpack('>3I', datas[offset:offlen])
        result = [ None, None, None ]

        offset = offlen
        offlen += ilen

        for i in range(pcount):
            dlen = struct.unpack('>I', datas[offset:offlen])

            offset = offlen
            offlen += dlen[0]

            result[i] = datas[offset:offlen]

            offset = offlen
            offlen += ilen

        return offset, result[0], result[1], result[2]

    def _process_read(self, list_r=[]):
        for sock in list_r:
            if sock is self.listen_fd:
                continue

            try:
                data = sock.recv(4096)

                if data:
                    buffer = self.client_buffer[sock]

                    buffer.extend(data)

                    while self._packet_available(buffer):
                        _length, _action, _data, _filedata = self._packet_parse(buffer)

                        self.client_buffer[sock] = buffer[_length:]

                        if self._process_login(sock, _action, _data):
                            buffer = self.client_buffer[sock]

                            continue

                        if self.event_handler:
                            self.event_handler(self, _action, _data, _filedata)

                        buffer = self.client_buffer[sock]

                    #if sock not in self.outputs:
                    #    self.outputs.append(sock)

                else:
                    traceback.print_exc()

                    self._process_except([ sock ])

            except:
                traceback.print_exc()

                self._process_except([ sock ])

    def _process_write(self, list_w=[]):
        for sock in list_w:
            self.outputs.remove(sock)

    def _process_except(self, list_e=[]):
        for sock in list_e:
            name = ''

            if sock in self.inputs:
                self.inputs.remove(sock)

            if sock in self.outputs:
                self.outputs.remove(sock)

            if sock in self.client_buffer:
                self.client_buffer[sock].clear()

                del self.client_buffer[sock]

            if sock in self.all_clients:
                name = self.all_clients[sock]

                del self.all_clients[sock]

            if name in self.named_clients:
                del self.named_clients[name]

            try:
                sock.close()

            except:
                traceback.print_exc()

    def _process_login(self, sock, action, data):
        if action == b'LOGIN':
            username = str(data, "utf-8")

            if self.all_clients[sock] != username:
                print('==> Bind user:', username, '->', sock)

            self.all_clients[sock] = username
            self.named_clients[username] = sock

            return True

        return False


def BaseTunnelServerEventHandler(tunnel, action, data, filedata):
    print('BaseTunnelServerEventHandler', action, data, len(filedata) if filedata else 0)

if __name__=='__main__':
    server = TunnelServer()

    server.run('localhost', 10021, BaseTunnelServerEventHandler)
