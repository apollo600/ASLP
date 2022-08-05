package com.tawht.kaylor.aslp_demo;

import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

/**
 * Created by ZhouMeng on 2018/8/31.
 * 将pcm文件转化为wav文件
 * pcm是无损wav文件中音频数据的一种编码方式，pcm加上wav文件头就可以转为wav格式，但wav还可以用其它方式编码。
 * 此类就是通过给pcm加上wav的文件头，来转为wav格式
 */
public class PcmToWav {
    /**
     * 合并多个pcm文件为一个wav文件
     * @param filePathList    pcm文件路径集合
     * @param destinationPath 目标wav文件路径
     * @return true|false
     */
    public static boolean mergePCMFilesToWAVFile(List<String> filePathList, String destinationPath) {
        File[] file = new File[filePathList.size()];
        byte[] buffer;

        int TOTAL_SIZE = 0;
        int fileNum = filePathList.size();

        for (int i = 0; i < fileNum; i++) {
            file[i] = new File(filePathList.get(i));
            TOTAL_SIZE += file[i].length();
        }

        // 填入参数，比特率等等。这里用的是16位单声道 16000 hz
        WaveHeader header = new WaveHeader();
        // 长度字段 = 内容的大小（TOTAL_SIZE) + 头部字段的大小(不包括前面4字节的标识符RIFF以及fileLength本身的4字节)
        header.fileLength = TOTAL_SIZE + (44 - 8);
        header.FmtHdrLeth = 16;
        header.BitsPerSample = 16;
        header.Channels = 1;
        header.FormatTag = 0x0001;
        header.SamplesPerSec = 16000;
        header.BlockAlign = (short) (header.Channels * header.BitsPerSample / 8);
        header.AvgBytesPerSec = header.BlockAlign * header.SamplesPerSec;
        header.DataHdrLeth = TOTAL_SIZE;

        byte[] h;
        try {
            h = header.getHeader();
        } catch (IOException e1) {
            Log.e("PcmToWav", e1.getMessage());
            return false;
        }

        // WAV标准，头部应该是44字节,如果不是44个字节则不进行转换文件
        if (h.length != 44) {
            return false;
        }

        //先删除目标文件
        File destFile = new File(destinationPath);
        if (destFile.exists()) {
            destFile.delete();
        }

        //合成所有的pcm文件的数据，写到目标文件
        try {
            buffer = new byte[1024 * 4]; // Length of All Files, Total Size
            InputStream inStream;
            OutputStream ouStream;

            ouStream = new BufferedOutputStream(new FileOutputStream(
                    destinationPath));
            ouStream.write(h, 0, h.length);
            for (int j = 0; j < fileNum; j++) {
                inStream = new BufferedInputStream(new FileInputStream(file[j]));
                int size = inStream.read(buffer);
                while (size != -1) {
                    ouStream.write(buffer);
                    size = inStream.read(buffer);
                }
                inStream.close();
            }
            ouStream.close();
        } catch (IOException ioe) {
            ioe.getMessage();
            return false;
        }

        return true;
    }
}

class WaveHeader {
    public final char[] fileID = {'R', 'I', 'F', 'F'};
    public int fileLength;
    public char[] wavTag = {'W', 'A', 'V', 'E'};
    public char[] FmtHdrID = {'f', 'm', 't', ' '};
    public int FmtHdrLeth;
    public short FormatTag;
    public short Channels;
    public int SamplesPerSec;
    public int AvgBytesPerSec;
    public short BlockAlign;
    public short BitsPerSample;
    public char[] DataHdrID = {'d','a','t','a'};
    public int DataHdrLeth;

    public byte[] getHeader() throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        WriteChar(bos, fileID);
        WriteInt(bos, fileLength);
        WriteChar(bos, wavTag);
        WriteChar(bos, FmtHdrID);
        WriteInt(bos,FmtHdrLeth);
        WriteShort(bos,FormatTag);
        WriteShort(bos,Channels);
        WriteInt(bos,SamplesPerSec);
        WriteInt(bos,AvgBytesPerSec);
        WriteShort(bos,BlockAlign);
        WriteShort(bos,BitsPerSample);
        WriteChar(bos,DataHdrID);
        WriteInt(bos,DataHdrLeth);
        bos.flush();
        byte[] r = bos.toByteArray();
        bos.close();
        return r;
    }

    private void WriteShort(ByteArrayOutputStream bos, int s) throws IOException {
        byte[] myByte = new byte[2];
        myByte[1] =(byte)( (s << 16) >> 24 );
        myByte[0] =(byte)( (s << 24) >> 24 );
        bos.write(myByte);
    }

    private void WriteInt(ByteArrayOutputStream bos, int n) throws IOException {
        byte[] buf = new byte[4];
        buf[3] =(byte)( n >> 24 );
        buf[2] =(byte)( (n << 8) >> 24 );
        buf[1] =(byte)( (n << 16) >> 24 );
        buf[0] =(byte)( (n << 24) >> 24 );
        bos.write(buf);
    }

    private void WriteChar(ByteArrayOutputStream bos, char[] id) {
        for (char c : id) {
            bos.write(c);
        }
    }
}
