package com.irhammuch.android.facerecognition;

import android.app.Application;
import android.util.Log;

import java.net.URISyntaxException;

import io.socket.client.IO;
import io.socket.client.Socket;

public class ClientSocket extends Application {
    private final String BASE_URL = "http://192.168.5.201:8080/";
    private Socket mSocket;

    public ClientSocket(String MAC) {
        try {
            // Append MAC address to the base URL
            String URL = BASE_URL + "?MAC=" + MAC;
            mSocket = IO.socket(URL);
            Log.d("CLIENTSOCKET", "Socket initializing");
        } catch (URISyntaxException e) {
            Log.e("CLIENTSOCKET", "Socket initializing failed");
            e.printStackTrace();
        }
    }

    public Socket getSocket() {
        return mSocket;
    }
}
