package com.irhammuch.android.facerecognition;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.usb.UsbDevice;
import android.media.Image;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.health.SystemHealthManager;
import android.renderscript.ScriptGroup;
import android.text.InputType;
import android.util.Base64;
import android.util.Log;
import android.util.Pair;
import android.view.Gravity;
import android.view.Surface;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.datatransport.runtime.retries.Retries;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import com.google.mlkit.vision.facemesh.FaceMesh;
import com.google.mlkit.vision.facemesh.FaceMeshDetection;
import com.google.mlkit.vision.facemesh.FaceMeshDetector;
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions;
import com.serenegiant.common.BaseActivity;
import com.serenegiant.glutils.AbstractRendererHolder;
import com.serenegiant.glutils.EGLBase;
import com.serenegiant.glutils.RenderHolderCallback;
import com.serenegiant.usb.CameraDialog;
import com.serenegiant.usb.IFrameCallback;
import com.serenegiant.usb.USBMonitor;
import com.serenegiant.usb.USBMonitor.OnDeviceConnectListener;
import com.serenegiant.usb.USBMonitor.UsbControlBlock;
import com.serenegiant.usb.UVCCamera;
import com.serenegiant.usbcameracommon.UVCCameraHandler;
import com.serenegiant.widget.CameraViewInterface;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.nio.channels.FileChannel;
import java.text.Format;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import javax.net.ssl.HttpsURLConnection;

import okhttp3.*;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_CODE = 1001;
    private static final String CAMERA_PERMISSION = Manifest.permission.CAMERA;
    private ConstraintLayout cameraLayout;
    private ImageView imageViewResult;
    private ImageButton btnFaceDetection;
    private ImageButton btnQRDetection;
    private ImageButton btnCardDetection;
    private ImageView imageViewSOICT;
    private PreviewView previewView;
    private CameraSelector cameraSelector;
    private ProcessCameraProvider cameraProvider;
    private int lensFacing = CameraSelector.LENS_FACING_BACK;
    private Preview previewUseCase;
    private ImageAnalysis analysisUseCase;
    private GraphicOverlay graphicOverlay;
    private ImageView previewImg;
    private TextView detectionTextView;

    private final HashMap<String, SimilarityClassifier.Recognition> registered = new HashMap<>(); //saved Faces
    private Interpreter tfLite;
    private boolean flipX = false;
    private boolean start = true;
    private float[][] embeddings;

    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    private static final int INPUT_SIZE = 112;
    private static final int OUTPUT_SIZE = 192;

    private static final float[] BANDWIDTH_FACTORS = {0.7f, 0.3f};
    private USBMonitor mUSBMonitor;
    private UVCCameraHandler mHandlerL;
    private CameraViewInterface mUVCCameraViewL;
    private Surface mLeftPreviewSurface;
    private View cameraLeftLayout = null;
    private UVCCameraHandler mHandlerR;
    private CameraViewInterface mUVCCameraViewR;
    private Surface mRightPreviewSurface;
    private View cameraRightLayout = null;
    private long lastUploadMillis = 0;
    private long lastCAM1Millis = 0;
    private long lastCAM2Millis = 0;
    private long lastFaceDetectedTime = 0;
    private boolean firstTime = true;
    private boolean isBusy = false;
    private boolean isWorking = false;
    private byte[] irData = null;
    Bitmap irBitmap = null;
    InputImage irInputImage = null;
    Bitmap rgbBitmap = null;
    Lock recognizingLock = new ReentrantLock();

    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            hideSystemUI();
        }
    }

    private void hideSystemUI() {
        View decorView = getWindow().getDecorView();
        decorView.setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        // Hide the nav bar and status bar
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN);
    }

    private void setupUVCCamera() {
        try {
            mUVCCameraViewL = (CameraViewInterface) findViewById(R.id.camera_view_L);
            mUVCCameraViewL.setAspectRatio(4.0 / 3.0);
            mHandlerL = UVCCameraHandler.createHandler(this, mUVCCameraViewL, 2, UVCCamera.DEFAULT_LPREVIEW_WIDTH, UVCCamera.DEFAULT_LPREVIEW_HEIGHT, UVCCamera.FRAME_FORMAT_MJPEG, BANDWIDTH_FACTORS[0]);

            mUVCCameraViewR = (CameraViewInterface) findViewById(R.id.camera_view_R);
            mUVCCameraViewR.setAspectRatio(4.0 / 3.0);
            mHandlerR = UVCCameraHandler.createHandler(this, mUVCCameraViewR, 2, UVCCamera.DEFAULT_RPREVIEW_WIDTH, UVCCamera.DEFAULT_RPREVIEW_HEIGHT, UVCCamera.FRAME_FORMAT_MJPEG, BANDWIDTH_FACTORS[1]);
            mUSBMonitor = new USBMonitor(this, mOnDeviceConnectListener);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public class RequestCameraUSBTimerTask extends TimerTask {
        private UsbDevice device;
        private USBMonitor monitor;
        private Timer timer;
        private Handler handler;

        public RequestCameraUSBTimerTask(USBMonitor _monitor, UsbDevice _device, Timer _timer, Handler _handler) {
            this.device = _device;
            this.monitor = _monitor;
            this.timer = _timer;
            this.handler = _handler;
        }

        @Override
        public void run() {
            if (mUVCCameraViewL.getSurfaceTexture() != null || mUVCCameraViewR.getSurfaceTexture() != null) {
                try {
                    if (timer != null) {
                        cancel();
                        timer = null;
                        RequestUSBRunnable runnable = new RequestUSBRunnable(monitor, device);
                        handler.postDelayed(runnable, 1000);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public class RequestUSBRunnable implements Runnable {
        private UsbDevice device;
        private USBMonitor monitor;

        public RequestUSBRunnable(USBMonitor _monitor, UsbDevice _device) {
            this.device = _device;
            this.monitor = _monitor;
        }

        @Override
        public void run() {
            try {
                monitor.requestPermission(device);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public class RequestRightCameraUSBTimerTask extends TimerTask {
        private UsbDevice device;
        private USBMonitor monitor;
        private Timer timer;
        private Handler handler;

        public RequestRightCameraUSBTimerTask(USBMonitor _monitor, UsbDevice _device, Timer _timer, Handler _handler) {
            this.device = _device;
            this.monitor = _monitor;
            this.timer = _timer;
            this.handler = _handler;
        }

        @Override
        public void run() {
            if (mUVCCameraViewR.getSurfaceTexture() != null) {
                try {
                    if (timer != null) {
                        cancel();
                        timer = null;
                        RequestUSBRunnable runnable = new RequestUSBRunnable(monitor, device);
                        handler.postDelayed(runnable, 1000);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private final USBMonitor.OnDeviceConnectListener mOnDeviceConnectListener = new USBMonitor.OnDeviceConnectListener() {

        @Override
        public void onAttach(UsbDevice device) {
            Log.d("MYUSB", "onAttach:" + device.toString());
            if (device.getProductName().equals("RGB-Camera")) {
                try {
                    if (mHandlerL.isOpened()) {
                        mHandlerL.stopPreview();
                        mHandlerL.close();
                    }
                } catch (Exception e) {
                    Log.d("MYUSB", "except: " + e.toString());
                }

                if (firstTime) {
                    firstTime = false;
                    Handler handler = new Handler();
                    Timer requestTimer = new Timer();
                    RequestCameraUSBTimerTask requestTimerTask = new RequestCameraUSBTimerTask(mUSBMonitor, device, requestTimer, handler);
                    requestTimer.schedule(requestTimerTask, 1000, 1000);
                } else {
                    mUSBMonitor.requestPermission(device);
                }
            }
            if (device.getProductName().equals("IR_Camera")) {
                try {
                    if (mHandlerR.isOpened()) {
                        mHandlerR.stopPreview();
                        mHandlerR.close();
                    }

                    if (firstTime) {
                        firstTime = false;
                        Handler handler = new Handler();
                        Timer requestTimer = new Timer();
                        RequestCameraUSBTimerTask requestTimerTask = new RequestCameraUSBTimerTask(mUSBMonitor, device, requestTimer, handler);
                        requestTimer.schedule(requestTimerTask, 1000, 1000);
                    } else {
                        mUSBMonitor.requestPermission(device);
                    }
                } catch (Exception e) {

                }
            }
        }

        @Override
        public void onDettach(UsbDevice device) {
            Log.d("MYUSB", "onDettach:" + device.toString());
        }

        @Override
        public void onConnect(UsbDevice device, USBMonitor.UsbControlBlock ctrlBlock, boolean createNew) {
            Log.d("MYUSB", "onConnect:" + device.toString());
            try {
                if (!mHandlerL.isOpened() && device.getProductName().equals("RGB-Camera")) {
                    Log.d("MYUSB", "onConnectL:" + device.toString());
                    mHandlerL.open(ctrlBlock);

                    Timer timer = new Timer();
                    timer.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            if (mHandlerL.isPreviewing()) {
                                timer.cancel();
                                Log.v("MYUSB", "RGB is previewing");
                                mHandlerL.setFrameCallback(new IFrameCallback() {
                                    @Override
                                    public void onFrame(ByteBuffer frame) {
                                        try {
                                            if (isWorking == false) {
                                                mHandlerL.mShowCamera = 0;
                                                mHandlerL.changePreviewSetting();
                                            }
                                            if (lastCAM1Millis == 0) {
                                                lastCAM1Millis = System.currentTimeMillis();
                                            }
                                            if (System.currentTimeMillis() - lastCAM1Millis < 500 || isBusy || isWorking == false) {
                                                return;
                                            }
                                            Log.v("MYUSB", "onFrameL");
                                            lastCAM1Millis = System.currentTimeMillis();
                                            byte[] data = new byte[frame.capacity()];
                                            frame.get(data, 0, data.length);
                                            analyze(data);

                                            if (System.currentTimeMillis() - lastFaceDetectedTime > 10000) {
                                                isWorking = false;
                                                runOnUiThread(new Runnable() {
                                                    @Override
                                                    public void run() {
                                                        btnFaceDetection.setVisibility(View.VISIBLE);
                                                        btnCardDetection.setVisibility(View.VISIBLE);
                                                        btnQRDetection.setVisibility(View.VISIBLE);
                                                        imageViewSOICT.setVisibility(View.VISIBLE);
                                                        mHandlerL.mShowCamera = 0;
                                                        mHandlerR.mShowCamera = 0;
                                                        mHandlerL.changePreviewSetting();
                                                        mHandlerR.changePreviewSetting();
                                                    }
                                                });
                                            }
                                        } catch (Exception ex) {
                                            ex.printStackTrace();
                                        }
                                    }
                                }, UVCCamera.FRAME_FORMAT_YUYV);
                            }
                        }
                    }, 0, 1000);
                } else if (!mHandlerR.isOpened() && device.getProductName().equals("IR_Camera")) {
                    Log.d("MYUSB", "onConnectR:" + device.toString());
                    mHandlerR.open(ctrlBlock);
                    Timer timer = new Timer();
                    timer.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            if (mHandlerR.isPreviewing()) {
                                timer.cancel();
                                Log.v("MYUSB", "IR is previewing");
                                mHandlerR.setFrameCallback(new IFrameCallback() {
                                    @Override
                                    public void onFrame(ByteBuffer frame) {
                                        try {
                                            if (isWorking == false) {
                                                mHandlerR.mShowCamera = 0;
                                                mHandlerR.changePreviewSetting();
                                            }
                                            if (lastCAM2Millis == 0) {
                                                lastCAM2Millis = System.currentTimeMillis();
                                            }
                                            if (System.currentTimeMillis() - lastCAM2Millis < 100 || isWorking == false) {
                                                return;
                                            }
                                            Log.v("MYUSB", "onFrameR");
                                            lastCAM2Millis = System.currentTimeMillis();
                                            irData = new byte[frame.capacity()];
                                            frame.get(irData, 0, irData.length);
                                        } catch (Exception ex) {
                                            ex.printStackTrace();
                                        }
                                    }
                                }, UVCCamera.FRAME_FORMAT_YUYV);
                            }
                        }
                    }, 0, 1000);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        @Override
        public void onDisconnect(UsbDevice device, USBMonitor.UsbControlBlock ctrlBlock) {
            Log.v("MYUSB", "onDisconnect:" + device.toString());
        }

        @Override
        public void onCancel(UsbDevice device) {
            Log.v("MYUSB", "onCancel:" + device.toString());
        }
    };

    private void scanDownloadFolder() {
        String path = "/storage/emulated/0/Download/";
        File directory = new File(path);
        File[] files = directory.listFiles();
        for (int i = 0; i < files.length; i++) {
            if (files[i].isFile()) {
                try {
                    Log.d("LOAD FACES", "(" + i + "/" + files.length + ")" + files[i].getAbsolutePath());
                    File f = new File("/storage/emulated/0/Download/Embeddings/" + files[i].getName());
                    if (f.exists()) {
                        try {
                            embeddings = new float[1][OUTPUT_SIZE];
                            FileInputStream fis = new FileInputStream(f);
                            DataInputStream dis = new DataInputStream(fis);
                            for (int k = 0; k < OUTPUT_SIZE; k++) {
                                embeddings[0][k] = dis.readFloat();
                            }
                            dis.close();
                            fis.close();
                            SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition("0", "", -1f);
                            result.setExtra(embeddings);
                            registered.put(files[i].getName(), result);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    } else {
                        Bitmap bm = BitmapFactory.decodeFile(files[i].getAbsolutePath());
                        analyze(bm, files[i].getName());
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraLayout = findViewById(R.id.camera_container);
        imageViewResult = findViewById(R.id.imageViewResult);
        previewView = findViewById(R.id.previewView);
        previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER);
        graphicOverlay = findViewById(R.id.graphic_overlay);
        previewImg = findViewById(R.id.preview_img);
        detectionTextView = findViewById(R.id.detection_text);
        imageViewSOICT = findViewById(R.id.imageViewSOICT);
        btnCardDetection = (ImageButton) findViewById(R.id.btn_card);
        btnQRDetection = (ImageButton) findViewById(R.id.btn_qr);
        btnFaceDetection = (ImageButton) findViewById(R.id.btn_face);
        btnFaceDetection.setFocusable(false);
        btnFaceDetection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mHandlerL.isPreviewing() == false) {
                    final SurfaceTexture st = mUVCCameraViewL.getSurfaceTexture();
                    Timer previewTimer = new Timer();
                    previewTimer.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            try {
                                Log.v("MYUSB", "RGB Preview Starting...");
                                previewTimer.cancel();
                                mHandlerL.mShowCamera = 1;
                                mHandlerL.startPreview(new Surface(st));

                                Log.v("MYUSB", "RGB Preview Started");
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }, 0, 1000);
                }

                if (mHandlerR.isPreviewing() == false) {
                    final SurfaceTexture st = mUVCCameraViewR.getSurfaceTexture();
                    Timer previewTimer = new Timer();
                    previewTimer.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            try {
                                previewTimer.cancel();
                                mHandlerR.mShowCamera = 1;
                                Log.v("MYUSB", "IR Preview Starting...");
                                mHandlerR.startPreview(new Surface(st));
                                Log.v("MYUSB", "IR Preview Started");
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }, 0, 1000);
                }

                lastFaceDetectedTime = System.currentTimeMillis();
                btnFaceDetection.setVisibility(View.INVISIBLE);
                btnCardDetection.setVisibility(View.INVISIBLE);
                btnQRDetection.setVisibility(View.INVISIBLE);
                imageViewSOICT.setVisibility(View.INVISIBLE);
                cameraLayout.setVisibility(View.VISIBLE);
                mHandlerL.mShowCamera = 1;
                mHandlerR.mShowCamera = 1;
                mHandlerL.changePreviewSetting();
                mHandlerR.changePreviewSetting();
                isWorking = true;
            }
        });

        //ImageButton addBtn = findViewById(R.id.add_btn);
        //addBtn.setOnClickListener((v -> addFace()));

        //ImageButton switchCamBtn = findViewById(R.id.switch_camera);
        //switchCamBtn.setOnClickListener((view -> switchCamera()));

        loadModel();

        scanDownloadFolder();

        setupUVCCamera();

    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onStart() {
        super.onStart();

        mUSBMonitor.register();

        if (mUVCCameraViewR != null)
            mUVCCameraViewR.onResume();
        if (mUVCCameraViewL != null)
            mUVCCameraViewL.onResume();
    }

    @Override
    protected void onStop() {
        mHandlerR.close();
        if (mUVCCameraViewR != null)
            mUVCCameraViewR.onPause();
        mHandlerL.close();
        if (mUVCCameraViewL != null)
            mUVCCameraViewL.onPause();
        mUSBMonitor.unregister();
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        if (mHandlerR != null) {
            mHandlerR = null;
        }
        if (mHandlerL != null) {
            mHandlerL = null;
        }
        if (mUSBMonitor != null) {
            mUSBMonitor.destroy();
            mUSBMonitor = null;
        }
        mUVCCameraViewR = null;
        mUVCCameraViewL = null;
        super.onDestroy();
    }

    /**
     * Permissions Handler
     */
    private void getPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{CAMERA_PERMISSION}, PERMISSION_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, int[] grantResults) {
        for (int r : grantResults) {
            if (r == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
                return;
            }
        }

        if (requestCode == PERMISSION_CODE) {
            setupCamera();
        }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    /**
     * Setup camera & use cases
     */
    private void startCamera() {
        if (ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) == PackageManager.PERMISSION_GRANTED) {
            setupCamera();
        } else {
            getPermissions();
        }
    }

    private void setupCamera() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraSelector = new CameraSelector.Builder().requireLensFacing(lensFacing).build();

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindAllCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "cameraProviderFuture.addListener Error", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindAllCameraUseCases() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            bindPreviewUseCase();
            bindAnalysisUseCase();
        }
    }

    private void bindPreviewUseCase() {
        if (cameraProvider == null) {
            return;
        }

        if (previewUseCase != null) {
            cameraProvider.unbind(previewUseCase);
        }

        Preview.Builder builder = new Preview.Builder();
        builder.setTargetAspectRatio(AspectRatio.RATIO_4_3);
        builder.setTargetRotation(getRotation());

        previewUseCase = builder.build();
        previewUseCase.setSurfaceProvider(previewView.getSurfaceProvider());

        try {
            cameraProvider
                    .bindToLifecycle(this, cameraSelector, previewUseCase);
        } catch (Exception e) {
            Log.e(TAG, "Error when bind preview", e);
        }
    }

    private void bindAnalysisUseCase() {
        if (cameraProvider == null) {
            return;
        }

        if (analysisUseCase != null) {
            cameraProvider.unbind(analysisUseCase);
        }

        Executor cameraExecutor = Executors.newSingleThreadExecutor();

        ImageAnalysis.Builder builder = new ImageAnalysis.Builder();
        builder.setTargetAspectRatio(AspectRatio.RATIO_4_3);
        builder.setTargetRotation(getRotation());

        analysisUseCase = builder.build();
        analysisUseCase.setAnalyzer(cameraExecutor, this::analyze);

        try {
            cameraProvider
                    .bindToLifecycle(this, cameraSelector, analysisUseCase);
        } catch (Exception e) {
            Log.e(TAG, "Error when bind analysis", e);
        }
    }

    protected int getRotation() throws NullPointerException {
        return previewView.getDisplay().getRotation();
    }

    private void switchCamera() {
        if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            lensFacing = CameraSelector.LENS_FACING_FRONT;
            flipX = true;
        } else {
            lensFacing = CameraSelector.LENS_FACING_BACK;
            flipX = false;
        }

        if (cameraProvider != null) cameraProvider.unbindAll();
        startCamera();
    }

    /**
     * Face detection processor
     */
    @SuppressLint("UnsafeOptInUsageError")
    private void analyze(@NonNull ImageProxy image) {
        if (image.getImage() == null) return;

        InputImage inputImage = InputImage.fromMediaImage(
                image.getImage(),
                image.getImageInfo().getRotationDegrees()
        );

        FaceDetector faceDetector = FaceDetection.getClient();

        faceDetector.process(inputImage)
                .addOnSuccessListener(faces -> onSuccessListener(faces, inputImage))
                .addOnFailureListener(e -> Log.e(TAG, "Barcode process failure", e))
                .addOnCompleteListener(task -> image.close());
    }

    private void onSuccessListener(List<Face> faces, InputImage inputImage) {
        Rect boundingBox = null;
        String name = null;
        float scaleX = (float) previewView.getWidth() / (float) inputImage.getHeight();
        float scaleY = (float) previewView.getHeight() / (float) inputImage.getWidth();

        if (faces.size() > 0) {
            Log.d("MYUSB", "Face Detected");
            detectionTextView.setText(R.string.face_detected);
            // get first face detected
            Face face = faces.get(0);

            // get bounding box of face;
            boundingBox = face.getBoundingBox();

            // convert img to bitmap & crop img
            Bitmap bitmap = mediaImgToBmp(
                    inputImage.getMediaImage(),
                    inputImage.getRotationDegrees(),
                    boundingBox);

            if (start) name = recognizeImage(bitmap);
            if (name != null) detectionTextView.setText(name);
        } else {
            detectionTextView.setText(R.string.no_face_detected);
        }
    }

    private void analyze(byte[] data) {
        if (irData != null) {
            irBitmap = BitmapFactory.decodeByteArray(irData, 0, irData.length);
            irInputImage = InputImage.fromBitmap(irBitmap, 0);
            FaceMeshDetector irDetector = FaceMeshDetection.getClient(new FaceMeshDetectorOptions.Builder().setUseCase(FaceMeshDetectorOptions.BOUNDING_BOX_ONLY).build());
            irDetector.process(irInputImage).addOnSuccessListener(new OnSuccessListener<List<FaceMesh>>() {
                @Override
                public void onSuccess(List<FaceMesh> faceMeshes) {
                    if (faceMeshes.size() > 0) {
                        lastFaceDetectedTime = System.currentTimeMillis();
                        Log.d("MYUSB", "Real face detected");
                        FaceMesh face = faceMeshes.get(0);
                        Rect boundingBox = face.getBoundingBox();
                        boundingBox.left -= 32;
                        boundingBox.right += 32;
                        boundingBox.top -= 32;
                        boundingBox.bottom += 32;
                        Bitmap tmpBitmap = BitmapFactory.decodeByteArray(data, 0, data.length);
                        try {
                            rgbBitmap = Bitmap.createBitmap(tmpBitmap, boundingBox.left, boundingBox.top, boundingBox.width(), boundingBox.height());
                        } catch (Exception e) {
                            rgbBitmap = null;
                        }
                        if (rgbBitmap != null) {
                            InputImage inputImage = InputImage.fromBitmap(rgbBitmap, 0);
                            FaceMeshDetector defaultDetector = FaceMeshDetection.getClient(new FaceMeshDetectorOptions.Builder().setUseCase(FaceMeshDetectorOptions.BOUNDING_BOX_ONLY).build());
                            defaultDetector.process(inputImage).addOnSuccessListener(new OnSuccessListener<List<FaceMesh>>() {
                                @Override
                                public void onSuccess(List<FaceMesh> faceMeshes) {
                                    Rect boundingBox = null;
                                    String name = null;
                                    if (faceMeshes.size() > 0) {
                                        isBusy = true;
                                        detectionTextView.setText(R.string.face_detected);
                                        // get first face detected
                                        FaceMesh face = faceMeshes.get(0);

                                        // get bounding box of face;
                                        boundingBox = face.getBoundingBox();

                                        // crop img
                                        try {
                                            Bitmap bitmap = Bitmap.createBitmap(rgbBitmap, boundingBox.left, boundingBox.top, boundingBox.width(), boundingBox.height());
                                            if (start)
                                                name = recognizeImage(getResizedBitmap(bitmap));
                                            if (name != null) {
                                                Log.d("MYUSB", "FACE DETECTED: " + name);
                                                detectionTextView.setText(name);

                                                if (name.equals("unknown") == false) {
                                                    Toast t = Toast.makeText(getApplicationContext(),
                                                            name.split("\\.")[0], Toast.LENGTH_LONG);
                                                    t.setGravity(Gravity.CENTER, 50, 200);
                                                    t.show();
                                                    OpenDoor(rgbBitmap, name);
                                                }

                                            }
                                        } catch (Exception e) {
                                            e.printStackTrace();
                                        }
                                        isBusy = false;
                                    } else {
                                        detectionTextView.setText(R.string.no_face_detected);
                                    }
                                }
                            });
                        }
                    }
                }
            });
        }
    }

    private void onSuccessListener(List<Face> faces, Bitmap bm) {
        Rect boundingBox = null;
        String name = null;
        float scaleX = (float) previewView.getWidth() / (float) bm.getHeight();
        float scaleY = (float) previewView.getHeight() / (float) bm.getWidth();

        if (faces.size() > 0) {
            Log.d("MYUSB", "Face Detected");
            detectionTextView.setText(R.string.face_detected);
            // get first face detected
            Face face = faces.get(0);

            // get bounding box of face;
            boundingBox = face.getBoundingBox();

            // crop img
            try {
                Bitmap bitmap = Bitmap.createBitmap(bm, boundingBox.left, boundingBox.top, boundingBox.width(), boundingBox.height());
                if (start) name = recognizeImage(getResizedBitmap(bitmap));
                if (name != null) {
                    Log.d("MYUSB", "FACE DETECTED: " + name);
                    detectionTextView.setText(name);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            detectionTextView.setText(R.string.no_face_detected);
        }

        graphicOverlay.draw(boundingBox, scaleX, scaleY, name);
    }

    private void analyze(Bitmap bm, String name) {
        InputImage inputImage = InputImage.fromBitmap(bm, 0);
        FaceDetector faceDetector = FaceDetection.getClient();

        faceDetector.process(inputImage)
                .addOnSuccessListener(faces -> onSuccessListener(faces, bm, name))
                .addOnFailureListener(e -> Log.e(TAG, "Barcode process failure", e));
    }

    private void onSuccessListener(List<Face> faces, Bitmap bm, String name) {
        Rect boundingBox = null;
        float scaleX = (float) previewView.getWidth() / (float) bm.getHeight();
        float scaleY = (float) previewView.getHeight() / (float) bm.getWidth();

        if (faces.size() > 0) {
            Log.d("MYUSB", "Face Detected");
            detectionTextView.setText(R.string.face_detected);
            // get first face detected
            Face face = faces.get(0);

            // get bounding box of face;
            boundingBox = face.getBoundingBox();

            try {
                recognizingLock.lock();
                // crop bitmap
                Bitmap bitmap = Bitmap.createBitmap(bm, boundingBox.left, boundingBox.top, boundingBox.width(), boundingBox.height());
                recognizeImage(getResizedBitmap(bitmap));
                SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition("0", "", -1f);
                result.setExtra(embeddings);
                registered.put(name, result);

                File f = new File("/storage/emulated/0/Download/Embeddings/" + name);
                f.createNewFile();
                FileOutputStream fos = new FileOutputStream(f);
                DataOutputStream dos = new DataOutputStream(fos);
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    dos.writeFloat(embeddings[0][k]);
                }
                dos.close();
                fos.close();
                recognizingLock.unlock();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            detectionTextView.setText(R.string.no_face_detected);
        }
    }

    /**
     * Recognize Processor
     */
    private void addFace() {
        start = false;
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter Name");

        // Set up the input
        final EditText input = new EditText(this);

        input.setInputType(InputType.TYPE_CLASS_TEXT);
        input.setMaxWidth(200);
        builder.setView(input);

        // Set up the buttons
        builder.setPositiveButton("ADD", (dialog, which) -> {
            //Toast.makeText(context, input.getText().toString(), Toast.LENGTH_SHORT).show();

            //Create and Initialize new object with Face embeddings and Name.
            SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                    "0", "", -1f);
            result.setExtra(embeddings);

            registered.put(input.getText().toString(), result);
            start = true;

        });
        builder.setNegativeButton("Cancel", (dialog, which) -> {
            start = true;
            dialog.cancel();
        });

        builder.show();
    }

    public String recognizeImage(final Bitmap bitmap) {
        // set image to preview
        //previewImg.setImageBitmap(bitmap);

        //Create ByteBuffer to store normalized image

        ByteBuffer imgData = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);

        imgData.order(ByteOrder.nativeOrder());

        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

        //get pixel values from Bitmap to normalize
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        //imgData is input to our model
        Object[] inputArray = {imgData};

        Map<Integer, Object> outputMap = new HashMap<>();


        embeddings = new float[1][OUTPUT_SIZE]; //output of model will be stored in this variable

        outputMap.put(0, embeddings);

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap); //Run model


        float distance;

        //Compare new face with saved Faces.
        if (registered.size() > 0) {

            final Pair<String, Float> nearest = findNearest(embeddings[0]);//Find closest matching face

            if (nearest != null) {

                final String name = nearest.first;
                distance = nearest.second;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
//                        Toast t = Toast.makeText(getApplicationContext(), "" + distance, Toast.LENGTH_LONG);
//                        Toast t = Toast.makeText(getApplicationContext(), name.split("\\.")[0], Toast.LENGTH_LONG);
//                        t.show();
                    }
                });
//                Log.d("SCORE", "" + distance);
                if (distance < 1.1f) //If distance between Closest found face is more than 1.000 ,then output UNKNOWN face.
                {
                    Log.d("SCORE", "" + distance);
                    return name;
                } else
                    return "unknown";
            }
        }

        return null;
    }

    //Compare Faces by distance between face embeddings
    private Pair<String, Float> findNearest(float[] emb) {

        Pair<String, Float> ret = null;
        for (Map.Entry<String, SimilarityClassifier.Recognition> entry : registered.entrySet()) {

            final String name = entry.getKey();
            final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

            float distance = 0;
            for (int i = 0; i < emb.length; i++) {
                float diff = emb[i] - knownEmb[i];
                distance += diff * diff;
            }
            distance = (float) Math.sqrt(distance);
            if (ret == null || distance < ret.second) {
                ret = new Pair<>(name, distance);
            }
        }

        return ret;

    }

    /**
     * Bitmap Converter
     */
    private Bitmap mediaImgToBmp(Image image, int rotation, Rect boundingBox) {
        //Convert media image to Bitmap
        Bitmap frame_bmp = toBitmap(image);

        //Adjust orientation of Face
        Bitmap frame_bmp1 = rotateBitmap(frame_bmp, rotation, flipX);

        //Crop out bounding box from whole Bitmap(image)
        float padding = 0.0f;
        RectF adjustedBoundingBox = new RectF(
                boundingBox.left - padding,
                boundingBox.top - padding,
                boundingBox.right + padding,
                boundingBox.bottom + padding);
        Bitmap cropped_face = getCropBitmapByCPU(frame_bmp1, adjustedBoundingBox);

        //Resize bitmap to 112,112
        return getResizedBitmap(cropped_face);
    }

    private Bitmap getResizedBitmap(Bitmap bm) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) 112) / width;
        float scaleHeight = ((float) 112) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        bm.recycle();
        return resizedBitmap;
    }

    private static Bitmap getCropBitmapByCPU(Bitmap source, RectF cropRectF) {
        Bitmap resultBitmap = Bitmap.createBitmap((int) cropRectF.width(),
                (int) cropRectF.height(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(resultBitmap);

        // draw background
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        paint.setColor(Color.WHITE);
        canvas.drawRect(//from  w w  w. ja v  a  2s. c  om
                new RectF(0, 0, cropRectF.width(), cropRectF.height()),
                paint);

        Matrix matrix = new Matrix();
        matrix.postTranslate(-cropRectF.left, -cropRectF.top);

        canvas.drawBitmap(source, matrix, paint);

        if (source != null && !source.isRecycled()) {
            source.recycle();
        }

        return resultBitmap;
    }

    private static Bitmap rotateBitmap(
            Bitmap bitmap, int rotationDegrees, boolean flipX) {
        Matrix matrix = new Matrix();

        // Rotate the image back to straight.
        matrix.postRotate(rotationDegrees);

        // Mirror the image along the X or Y axis.
        matrix.postScale(flipX ? -1.0f : 1.0f, 1.0f);
        Bitmap rotatedBitmap =
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // Recycle the old bitmap if it has changed.
        if (rotatedBitmap != bitmap) {
            bitmap.recycle();
        }
        return rotatedBitmap;
    }

    private static byte[] YUV_420_888toNV21(Image image) {

        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * height / 4;

        byte[] nv21 = new byte[ySize + uvSize * 2];

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer(); // Y
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer(); // U
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer(); // V

        int rowStride = image.getPlanes()[0].getRowStride();
        assert (image.getPlanes()[0].getPixelStride() == 1);

        int pos = 0;

        if (rowStride == width) { // likely
            yBuffer.get(nv21, 0, ySize);
            pos += ySize;
        } else {
            long yBufferPos = -rowStride; // not an actual position
            for (; pos < ySize; pos += width) {
                yBufferPos += rowStride;
                yBuffer.position((int) yBufferPos);
                yBuffer.get(nv21, pos, width);
            }
        }

        rowStride = image.getPlanes()[2].getRowStride();
        int pixelStride = image.getPlanes()[2].getPixelStride();

        assert (rowStride == image.getPlanes()[1].getRowStride());
        assert (pixelStride == image.getPlanes()[1].getPixelStride());

        if (pixelStride == 2 && rowStride == width && uBuffer.get(0) == vBuffer.get(1)) {
            // maybe V an U planes overlap as per NV21, which means vBuffer[1] is alias of uBuffer[0]
            byte savePixel = vBuffer.get(1);
            try {
                vBuffer.put(1, (byte) ~savePixel);
                if (uBuffer.get(0) == (byte) ~savePixel) {
                    vBuffer.put(1, savePixel);
                    vBuffer.position(0);
                    uBuffer.position(0);
                    vBuffer.get(nv21, ySize, 1);
                    uBuffer.get(nv21, ySize + 1, uBuffer.remaining());

                    return nv21; // shortcut
                }
            } catch (ReadOnlyBufferException ex) {
                // unfortunately, we cannot check if vBuffer and uBuffer overlap
            }

            // unfortunately, the check failed. We must save U and V pixel by pixel
            vBuffer.put(1, savePixel);
        }

        // other optimizations could check if (pixelStride == 1) or (pixelStride == 2),
        // but performance gain would be less significant

        for (int row = 0; row < height / 2; row++) {
            for (int col = 0; col < width / 2; col++) {
                int vuPos = col * pixelStride + row * rowStride;
                nv21[pos++] = vBuffer.get(vuPos);
                nv21[pos++] = uBuffer.get(vuPos);
            }
        }

        return nv21;
    }

    private Bitmap toBitmap(Image image) {

        byte[] nv21 = YUV_420_888toNV21(image);


        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    /**
     * Model loader
     */
    @SuppressWarnings("deprecation")
    private void loadModel() {
        try {
            //model name
            String modelFile = "mobile_face_net.tflite";
            tfLite = new Interpreter(loadModelFile(MainActivity.this, modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void OpenDoor(Bitmap cameraBM, String name) {
        try {
            isWorking = false;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    try {
                        mHandlerL.mShowCamera = 0;
                        mHandlerR.mShowCamera = 0;
                        mHandlerL.changePreviewSetting();
                        mHandlerR.changePreviewSetting();
                        cameraLayout.setVisibility(View.INVISIBLE);
                        imageViewSOICT.setVisibility(View.VISIBLE);
                        File f = new File("/storage/emulated/0/Download/" + name);
                        if (f.exists()) {
                            imageViewResult.setVisibility(View.VISIBLE);
                            Bitmap bm = BitmapFactory.decodeFile("/storage/emulated/0/Download/" + name);
                            imageViewResult.setImageBitmap(bm);

                            Thread uploadThread = new Thread(new Runnable() {
                                @Override
                                public void run() {
                                    Log.d("TEST", "Uploading");
                                    try {
                                        if (System.currentTimeMillis() - lastUploadMillis > 5000) {
                                            lastUploadMillis = System.currentTimeMillis();
                                            ByteArrayOutputStream imageByteStream = new ByteArrayOutputStream();
                                            cameraBM.compress(Bitmap.CompressFormat.JPEG, 70, imageByteStream);

                                            byte[] imageData = imageByteStream.toByteArray();
                                            imageByteStream.close();
                                            File f = new File("/storage/emulated/0/Download/temp/TEMPFILE.JPEG");
                                            try (FileOutputStream fos = new FileOutputStream(f)) {
                                                fos.write(imageData);
                                                Log.d("TEST WRITE", "Image written to file");
                                            } catch (IOException e) {
                                                Log.d("TEST WRITE ERR", "Image written to file failed" + e.getMessage());
                                            }

                                            String macAddress = getMacAddress(getApplicationContext());
                                            Log.d("MAC ADDRESS", macAddress);

                                            String url = "http://192.168.5.245:8080/guest/newEntry";

                                            // Create a client
                                            OkHttpClient client = new OkHttpClient();

                                            // Create a multipart body builder
                                            MultipartBody.Builder requestBodyBuilder = new MultipartBody.Builder()
                                                    .setType(MultipartBody.FORM)
                                                    .addFormDataPart("File", "TEMPFILE.JPEG",
                                                            RequestBody.create(MediaType.parse("image/jpeg"), f))
                                                    .addFormDataPart("MAC", macAddress)
                                                    .addFormDataPart("name", name.split("\\.")[0]);

                                            // Build the request body
                                            RequestBody requestBody = requestBodyBuilder.build();

                                            // Build the request
                                            Request request = new Request.Builder()
                                                    .url(url)
                                                    .post(requestBody)
                                                    .build();

                                            // Make the request asynchronously
                                            client.newCall(request).enqueue(new Callback() {
                                                @Override
                                                public void onResponse(Call call, Response response) throws IOException {
                                                    if (response.isSuccessful()) {
                                                        String responseBody = response.body().string();
                                                        Log.d("UPLOADCAM", responseBody);
                                                    } else {
                                                        Log.d("UPLOADCAM", "Upload failed");
                                                    }
                                                }

                                                @Override
                                                public void onFailure(Call call, IOException e) {
                                                    Log.d("UPLOADCAM", "Upload failed");
                                                }
                                            });


                                        }
                                    } catch (Exception e) {
                                        e.printStackTrace();
                                    }
                                }
                            });
                            uploadThread.start();

                            Handler handler = new Handler();
                            handler.postDelayed(new Runnable() {
                                @Override
                                public void run() {
                                    imageViewResult.setVisibility(View.INVISIBLE);
                                    btnFaceDetection.setVisibility(View.VISIBLE);
                                    btnCardDetection.setVisibility(View.VISIBLE);
                                    btnQRDetection.setVisibility(View.VISIBLE);
                                }
                            }, 5000);
                        }
                    } catch (Exception e) {

                    }
                }
            });

            FileOutputStream gpioStream = new FileOutputStream(new File("/sys/class/gpio/gpio63/value"));
            gpioStream.write(new byte[]{49});
            gpioStream.flush();
            gpioStream.close();
            Thread.sleep(500);
            gpioStream = new FileOutputStream(new File("/sys/class/gpio/gpio63/value"));
            gpioStream.write(new byte[]{48});
            gpioStream.flush();
            gpioStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String getMacAddress(Context context) {
        WifiManager wifiManager = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);

        if (wifiManager == null) {
            Log.e(TAG, "WifiManager is null. Cannot retrieve MAC address.");
            return null;
        }

        WifiInfo wifiInfo = wifiManager.getConnectionInfo();

        if (wifiInfo == null) {
            Log.e(TAG, "WifiInfo is null. Cannot retrieve MAC address.");
            return null;
        }

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            return null;
        }
        String macAddress = wifiInfo.getMacAddress();

        if (macAddress == null) {
            Log.e(TAG, "MAC address is null.");
            return null;
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && macAddress.equals("02:00:00:00:00:00")) {
            Log.e(TAG, "MAC address is not available in Android 6.0+.");
            return null;
        }

        return macAddress;
    }
}