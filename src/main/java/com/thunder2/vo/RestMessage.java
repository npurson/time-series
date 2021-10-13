package com.thunder2.vo;

import lombok.Data;

import java.io.Serializable;

@Data
public class RestMessage<T> implements Serializable {
    private static final long serialVersionUID = -1865510446859810360L;
    private String message;
    private Integer code;
    private T data;

    public RestMessage() {

    }

    public static RestMessage newInstance(Integer code, String message) {
        return new RestMessage(code, message);
    }

    public static <T> RestMessage<T> newInstance(Integer code, String msg, T data) {
        return new RestMessage<T>(code, msg, data);
    }

    public RestMessage(String message, T data) {
        this.message = message;
        this.data = data;
        this.code = 0;
    }

    public RestMessage(int code, String message) {
        this.code = code;
        this.message = message;
    }

    public RestMessage(Integer code, String message, T data) {
        this.message = message;
        this.data = data;
        this.code = code;
    }
}
