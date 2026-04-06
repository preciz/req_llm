defmodule ReqLLM.Providers.Minimax do
  @moduledoc """
  Minimax provider – OpenAI-compatible Chat Completions API.

  ## Implementation

  Uses built-in OpenAI-style encoding/decoding defaults.
  Minimax provides an OpenAI-compatible endpoint at `/v1/chat/completions`.

  ## Authentication

  Requires a Minimax API key from https://platform.minimaxi.com/

  ## Configuration

      # Add to .env file (automatically loaded)
      MINIMAX_API_KEY=your-api-key

  ## Examples

      # Basic usage
      ReqLLM.generate_text("minimax:minimax-text-01", "Hello!")

      # With custom parameters
      ReqLLM.generate_text("minimax:minimax-m2.7", "Write a function",
        temperature: 0.2,
        max_tokens: 2000
      )

      # Streaming
      ReqLLM.stream_text("minimax:minimax-text-01", "Tell me a story")
      |> Enum.each(&IO.write/1)
  """

  use ReqLLM.Provider,
    id: :minimax,
    default_base_url: "https://api.minimax.io/v1",
    default_env_key: "MINIMAX_API_KEY"

  use ReqLLM.Provider.Defaults

  @provider_schema [
    response_format: [
      type: {:or, [:map, :keyword_list]},
      doc: "Response format (e.g. %{type: \"json_schema\"})"
    ]
  ]

  @doc """
  Custom prepare_request for :object operations.
  Minimax models sometimes struggle with native tool_choice or strict response_format flags,
  so we enforce structured output via json_schema and an explicit system prompt.
  """
  @impl ReqLLM.Provider
  def prepare_request(:object, model_spec, prompt, opts) do
    provider_opts = Keyword.get(opts, :provider_options, [])
    compiled_schema = Keyword.fetch!(opts, :compiled_schema)

    json_schema_map = ReqLLM.Schema.to_json(compiled_schema.schema)

    json_schema_payload = %{
      type: "json_schema",
      json_schema: %{
        name: "structured_output",
        strict: true,
        schema: json_schema_map
      }
    }

    updated_provider_opts =
      provider_opts
      |> Keyword.put(:response_format, json_schema_payload)

    updated_opts =
      opts
      |> Keyword.put(:provider_options, updated_provider_opts)
      |> Keyword.delete(:tools)
      |> Keyword.delete(:tool_choice)

    instruction =
      "You must respond with ONLY valid JSON matching this schema: #{Jason.encode!(json_schema_map)}. Do NOT wrap the JSON in markdown blocks like ```json. Output ONLY the raw JSON object."

    instruction_msg = ReqLLM.Context.system(instruction)
    updated_prompt = %{prompt | messages: [instruction_msg | prompt.messages]}

    updated_opts =
      case Keyword.get(updated_opts, :max_tokens) do
        nil -> Keyword.put(updated_opts, :max_tokens, 4096)
        _tokens -> updated_opts
      end

    updated_opts = Keyword.put(updated_opts, :operation, :object)

    ReqLLM.Provider.Defaults.prepare_request(
      __MODULE__,
      :chat,
      model_spec,
      updated_prompt,
      updated_opts
    )
  end

  @impl ReqLLM.Provider
  def prepare_request(operation, model_spec, prompt, opts) do
    ReqLLM.Provider.Defaults.prepare_request(__MODULE__, operation, model_spec, prompt, opts)
  end
end
