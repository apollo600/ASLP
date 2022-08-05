package com.tawht.kaylor.aslp_demo;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private Button button, button2, button3, button4;
    private AudioRecord audioRecord;
    private Runnable denoisyRunnable;
    private boolean isRecording;
    private int minBufferSize;
    private final int AUDIO_SOURCE = MediaRecorder.AudioSource.MIC;
    private final int SAMPLE_RATE_INHZ = 16000;
    private final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private MediaPlayer mediaPlayer;
    private TextView textView, textView2;
    private final List<String> mPermissionList = new ArrayList<>();
    private final static int ACCESS_FINE_ERROR_CODE = 0x1213;
    private final ExecutorService singleThreadExecutor = Executors.newSingleThreadExecutor();
    private final static int DENOISY_CODE_0 = 0x0011;
    private final String filepcmPath = "/ASLP/pcm/";
    private final String filewavPath = "/ASLP/wav/";
    private final String fileName_mix = "mix";
    private final String fileName_mix_denoisy = "mix_denoisy";
    private File filepcm, filewav;
    private Timer timer;
    private TimerTask timerTask;
    private Module module;
    private Map<String, String> info;
    private int buffer_T = 1;
    private int flag = 0; //模式0：语音增强；
    private boolean newFileSystem = false;

    public MainActivity() {
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setPermissions(new String[]{
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE}
        );

        initView();
        initAudio();

        if (Build.VERSION.SDK_INT > 29) {
            newFileSystem = true;
        }
    }

    private void initView() {
        button = this.findViewById(R.id.button);
        button2 = this.findViewById(R.id.button2);
        button3 = this.findViewById(R.id.button3);
        button4 = this.findViewById(R.id.button4);

        textView = this.findViewById(R.id.textView);
        textView2 = this.findViewById(R.id.textView2);

        setClickListener(button, button2, button3, button4);
    }

    private void initAudio() {
        cleanASLP();

        minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE_INHZ, CHANNEL_CONFIG, AUDIO_FORMAT);
        mediaPlayer = new MediaPlayer();
        try {
            module = Module.load(assetFilePath(this, "model.pt"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        info = new HashMap<>();
        info.put("音频时长：", "尚未录音");
        info.put("当前模型：", "增强模型");
        info.put("buffer大小：", "1秒");
        info.put("增强完成模型：", "尚未生成");
        info.put("增强处理时间：", "尚未生成");
        info.put("参考音频时长：", "尚未录音");
        info.put("提取处理时间：", "尚未生成");

        showText();
    }

    private void setClickListener(View... views) {
        for (View view : views) {
            view.setOnClickListener(this);
        }
    }

    @SuppressLint("HandlerLeak")
    private final Handler handler = new Handler() {
        @Override
        public void handleMessage(Message message) {
            switch (message.what){
                case DENOISY_CODE_0:
                    Toast.makeText(getApplicationContext(),
                            "处理完毕，耗时："+long2Time((long)message.obj)+"秒",
                            Toast.LENGTH_LONG).show();
                    MainActivity.this.runOnUiThread(() -> {
                        info.put("增强完成模型：", info.get("当前模型："));
                        info.put("增强处理时间：", long2Time((long)message.obj)+"秒");
                        showText();
                        button.setEnabled(true);
                        button3.setEnabled(true);
                        button4.setEnabled(true);
                    });
                    break;
            }
        }
    };

    @SuppressLint({"NonConstantResourceId", "DefaultLocale"})
    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.button:
                if (!isRecording) {
                    button.setText(R.string.button_2);
                    button2.setEnabled(false);
                    button3.setEnabled(false);

                    RecordingStart(fileName_mix);
                } else {
                    long fileLength = RecordingPause(fileName_mix);

                    buffer_T = (int) (fileLength/SAMPLE_RATE_INHZ/2 + 1);
                    info.put("音频时长：",String.format("%.2f秒", (float)fileLength/SAMPLE_RATE_INHZ/2));
                    info.put("buffer大小：", String.format("%d秒", buffer_T));
                    showText();

                    button.setText(R.string.button);
                    button2.setEnabled(true);
                    button3.setEnabled(true);
                }
                break;

            case R.id.button2:
                if (newFileSystem) {
                    filewav = new File(getExternalFilesDir(null) + filewavPath + fileName_mix + ".wav");
                } else {
                    filewav = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                            filewavPath + fileName_mix + ".wav");
                }
                if (filewav.exists()) {
                    try {
                        mediaPlayer.reset();
                        mediaPlayer.setDataSource(filewav.getAbsolutePath());
                        mediaPlayer.prepare();
                        mediaPlayer.start();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    Toast.makeText(this, "请先录音", Toast.LENGTH_LONG).show();
                }
                filewav = null;
                break;

            case R.id.button3:
                if (flag == 0) {
                    if (newFileSystem) {
                        filepcm = new File(getExternalFilesDir(null) + filepcmPath + fileName_mix +".pcm");
                    } else {
                        filepcm = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                                filepcmPath + fileName_mix +".pcm");
                    }
//                    if (filepcm.exists() && module != null) {
                    if (filepcm.exists()) {
                        button.setEnabled(false);
                        button3.setEnabled(false);
                        button4.setEnabled(false);
                        Toast.makeText(this, "开始处理", Toast.LENGTH_LONG).show();
                        cleanASLP(fileName_mix_denoisy);

                        String filein, fileoutpcm, destinationPath;
                        File fileout;
                        if (newFileSystem) {
                            filein = getExternalFilesDir(null) + filepcmPath + fileName_mix + ".pcm";
                            fileout = new File(getExternalFilesDir(null) + filepcmPath + fileName_mix_denoisy + ".pcm");
                            fileoutpcm = getExternalFilesDir(null) + filepcmPath + fileName_mix_denoisy + ".pcm";
                            destinationPath = getExternalFilesDir(null) + filewavPath + fileName_mix_denoisy + ".wav";
                        } else {
                            filein = Environment.getExternalStorageDirectory().getAbsolutePath() +
                                    filepcmPath + fileName_mix +".pcm";
                            fileout = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                                    filepcmPath + fileName_mix_denoisy + ".pcm");
                            fileoutpcm = Environment.getExternalStorageDirectory().getAbsolutePath() +
                                    filepcmPath + fileName_mix_denoisy + ".pcm";
                            destinationPath = Environment.getExternalStorageDirectory().getAbsolutePath() +
                                    filewavPath + fileName_mix_denoisy + ".wav";
                        }
                        denoisyRunnable = () -> {
                            long startTime = System.currentTimeMillis();
                            denoisy(filein, fileout);
                            long endTime = System.currentTimeMillis();

                            List<String> filePathList = new ArrayList<>();
                            filePathList.add(fileoutpcm);
                            PcmToWav.mergePCMFilesToWAVFile(filePathList, destinationPath);

                            Message message = new Message();
                            message.what = DENOISY_CODE_0;
                            message.obj = endTime-startTime;
                            handler.sendMessage(message);
                        };
                        singleThreadExecutor.execute(denoisyRunnable);
                    } else {
                        if (!filepcm.exists()) {
                            Toast.makeText(this, "请先录音", Toast.LENGTH_LONG).show();
                        }
//                        if (module == null) {
//                            Toast.makeText(this, "模型为空", Toast.LENGTH_LONG).show();
//                        }
                    }
                    filepcm = null;
                }
                break;

            case R.id.button4:
                if (flag == 0) {
                    if (newFileSystem) {
                        filewav = new File(getExternalFilesDir(null) + filewavPath + fileName_mix_denoisy + ".wav");
                    } else {
                        filewav = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                                filewavPath + fileName_mix_denoisy + ".wav");
                    }
                }
                if (filewav.exists()) {
                    try {
                        mediaPlayer.reset();
                        mediaPlayer.setDataSource(filewav.getAbsolutePath());
                        mediaPlayer.prepare();
                        mediaPlayer.start();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    Toast.makeText(this, "请先增强处理", Toast.LENGTH_LONG).show();
                }
                filewav = null;
                break;
        }
    }

    @Override
    protected void onDestroy() {
        singleThreadExecutor.shutdownNow();
        module.destroy();
        audioRecord.stop();
        audioRecord.release();
        mediaPlayer.stop();
        mediaPlayer.release();
        super.onDestroy();
    }

    private void RecordingStart(String fileName) {
        if (audioRecord == null) {
            audioRecord = new AudioRecord(AUDIO_SOURCE, SAMPLE_RATE_INHZ, CHANNEL_CONFIG, AUDIO_FORMAT, minBufferSize);
        }
        cleanASLP(fileName);
        isRecording = true;

        Runnable audioRecordRunnable = () -> {
            try {
                File file;
                if (newFileSystem) {
                    file = new File(getExternalFilesDir(null) + filepcmPath + fileName +".pcm");
                } else {
                    file = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                            filepcmPath + fileName +".pcm");
                }
                FileOutputStream fos = new FileOutputStream(file);
                byte[] audioData = new byte[minBufferSize];
                while (isRecording) {
                    int readSize = audioRecord.read(audioData, 0, minBufferSize);
                    if (readSize != AudioRecord.ERROR_INVALID_OPERATION) {
                        fos.write(audioData);
                    }
                }
                fos.close();
                List<String> filePathList = new ArrayList<>();
                filePathList.add(file.getAbsolutePath());
                String destinationPath;
                if (newFileSystem) {
                    destinationPath = getExternalFilesDir(null) + filewavPath + fileName + ".wav";
                } else {
                    destinationPath = Environment.getExternalStorageDirectory().getAbsolutePath() +
                            filewavPath + fileName + ".wav";
                }
                PcmToWav.mergePCMFilesToWAVFile(filePathList, destinationPath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        };

        timer = new Timer();
        timerTask = new TimerTask() {
            final long begin = SystemClock.uptimeMillis();
            @Override
            public void run() {
                int time = (int) ((SystemClock.uptimeMillis() - begin) / 10);
                runOnUiThread(() -> textView.setText(int2Time(time)));
            }
        };
        timer.schedule(timerTask, 0, 10);
        audioRecord.startRecording();
        singleThreadExecutor.execute(audioRecordRunnable);
    }

    private long RecordingPause(String fileName) {
        isRecording = false;
        timerTask.cancel();
        timer.cancel();
        timerTask = null;
        timer = null;
        audioRecord.stop();
        File filepcm;
        if (newFileSystem) {
            filepcm = new File(getExternalFilesDir(null) + filepcmPath + fileName + ".pcm");
        } else {
            filepcm = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                    filepcmPath + fileName + ".pcm");
        }
        return filepcm.length();
    }

    private void setPermissions(String[] permissions) {
        mPermissionList.clear();
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                mPermissionList.add(permission);
            }
        }
        if (mPermissionList.isEmpty()) {
            Toast.makeText(this, "已经授权", Toast.LENGTH_LONG).show();
        } else {
            permissions = mPermissionList.toArray(new String[0]);
            ActivityCompat.requestPermissions(this, permissions, MainActivity.ACCESS_FINE_ERROR_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int i = 0; i < grantResults.length; i++) {
            boolean showRequestPermission = ActivityCompat.shouldShowRequestPermissionRationale(this, permissions[i]);
            if (showRequestPermission) {
                Toast.makeText(this, "权限未申请", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void showText() {
        String head = "信息\n";
        switch (flag) {
            case 0:
                head += "音频时长：" + info.get("音频时长：") + "\n";
                head += "当前模型：" + info.get("当前模型：") + "\n";
                head += "buffer大小："  + info.get("buffer大小：") + "\n";
                head += "增强完成模型：" + info.get("增强完成模型：") + "\n";
                head += "增强处理时间：" + info.get("增强处理时间：") + "\n";
                break;
        }
        textView2.setText(head);
    }

    @SuppressLint("DefaultLocale")
    private static String int2Time(int time) {
        int m = time/6000;
        int s = time/100%60;
        int ms= time%100;
        return String.format("%02d:%02d.%02d", m,s,ms);
    }

    @SuppressLint("DefaultLocale")
    private static String long2Time(long time) {
        return String.format("%d.%03d", time/1000, time%1000);
    }

    private void cleanASLP() {
        File folderpcm;
        File folderwav;

        if (newFileSystem) {
            folderpcm = new File(getExternalFilesDir(null) + filepcmPath);
            folderwav = new File(getExternalFilesDir(null) + filewavPath);
        } else {
            folderpcm = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                    filepcmPath);
            folderwav = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                    filewavPath);
        }

        if (folderpcm.exists()) {
            File[] files = folderpcm.listFiles();
            for (File f : files) {
                f.delete();
            }
        } else {
            folderpcm.mkdirs();
        }

        if (folderwav.exists()) {
            File[] files = folderwav.listFiles();
            for (File f : files) {
                f.delete();
            }
        } else {
            folderwav.mkdirs();
        }
    }

    private void cleanASLP(String fileName) {
        File folderpcm, folderwav, filepcm, filewav;

        if (newFileSystem) {
            folderpcm = new File(getExternalFilesDir(null) + filepcmPath);
            folderwav = new File(getExternalFilesDir(null) + filewavPath);
            filepcm = new File(getExternalFilesDir(null) + filepcmPath + fileName +".pcm");
            filewav = new File(getExternalFilesDir(null) + filewavPath + fileName +".wav");
        } else {
            folderpcm = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                    filepcmPath);
            folderwav = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                    filewavPath);
            filepcm = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                    filepcmPath + fileName +".pcm");
            filewav = new File(Environment.getExternalStorageDirectory().getAbsolutePath() +
                    filewavPath + fileName +".wav");
        }

        if (!folderpcm.exists()) {
            folderpcm.mkdirs();
        }
        if (!folderwav.exists()) {
            folderwav.mkdirs();
        }
        if (filepcm.exists()) {
            filepcm.delete();
        }
        if (filewav.exists()) {
            filewav.delete();
        }
    }

    private static short[] byte2short(byte[] bytes) {
        short[] shorts = new short[bytes.length/2];
        for (int i = 0; i < shorts.length; i++) {
            shorts[i] = (short) ((bytes[i*2+1]&0xff)<<8| (bytes[i*2]&0xff));
        }
        return shorts;
    }

    private static float[] normalize(short[] shorts) {
        float[] floats = new float[shorts.length];
        float max_normalize = Short.MAX_VALUE;
        for (int i = 0; i < floats.length; i++) {
            floats[i] = shorts[i];
            floats[i] /= max_normalize;
        }
        return floats;
    }

    private static short[] inormalize(float[] floats) {
        short[] shorts = new short[floats.length];
        float max_normalize = Short.MAX_VALUE;
        for (int i = 0; i < floats.length; i++) {
            shorts[i] = (short)(floats[i] * max_normalize);
        }
        return shorts;
    }

    private static byte[] short2byte(short[] shorts) {
        byte[] bytes = new byte[shorts.length*2];
        for (int i = 0; i < shorts.length; i++) {
            bytes[i*2+1] = (byte) ((shorts[i]>>8)&0xff);
            bytes[i*2] = (byte) (shorts[i]&0xff);
        }
        return bytes;
    }

    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private void denoisy(String fileinpath, File fileout) {
        File filein = new File(fileinpath);
        final int fs = SAMPLE_RATE_INHZ;
        int T = buffer_T;
        try {
            byte[] buffer = new byte[fs * T * 2];
            InputStream inStream = new BufferedInputStream(new FileInputStream(filein));
            OutputStream ouStream = new BufferedOutputStream(new FileOutputStream(fileout));
            int size = inStream.read(buffer);
            while (size != -1) {
                short[] input = byte2short(buffer);
                float[] input_float = normalize(input);

                long[] shape = new long[]{fs*T};
                final Tensor inputTensor = Tensor.fromBlob(input_float, shape);
                final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
                float[] output_model = outputTensor.getDataAsFloatArray();

                short[] output = inormalize(output_model);
                byte[] buffer_output = short2byte(output);

                if (size != fs * T * 2) {
                    ouStream.write(Arrays.copyOfRange(buffer_output, 0, size));
                } else {
                    ouStream.write(buffer_output);
                }
                size = inStream.read(buffer);
            }
            inStream.close();
            ouStream.close();
        } catch (IOException ioe) {
            ioe.getMessage();
        }
    }


}